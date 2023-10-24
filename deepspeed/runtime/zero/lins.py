import os
import sys
import math
from typing import Iterable, Dict
# from enum import Enum
from typing import List
import logging
import torch
from torch import Tensor
# from torch.nn import Module
from torch.nn import Parameter
from functools import partial

import deepspeed
import itertools
from deepspeed.utils import instrument_w_nvtx, logger, log_dist

from deepspeed import comm as dist
from deepspeed.runtime.zero.mics_utils import (create_mics_comm_groups, scale_tensors)
# from deepspeed.runtime.zero.parameter_offload import (is_zero_param)
from deepspeed.runtime.zero.partition_parameters import Init, AllGatherCoalescedHandle, ZeroParamStatus, free_param, print_rank_0, _no_gather_coalesced, _dist_allgather_fn, AllGatherHandle, assert_ints_same_as_other_ranks
from deepspeed.accelerator import get_accelerator
from ..swap_tensor.partitioned_param_swapper import PartitionedParamStatus
from ..utils import get_only_unique_item, see_memory_usage
from deepspeed.utils.debug import (debug_param2name_id_shape_device, debug_param2name_id_shape_status)
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3, reuse_buffers
from deepspeed.runtime.comm.coalesced_collectives import reduce_scatter_coalesced
from deepspeed.runtime.zero.lins_utils import zero35_g_p_reduce_scatter_coalesced, \
    zero35_g_p_all_gather_coalesced, zero35_debug, set_lins_parition_type, zero35_judge_gahter_boundary, global_parition_type, global_lins_utils, LinSProcessGroup

# Copyright Shanghai AI Laboratory, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

ENBALE_MEM_DEBUG = False
ENBALE_COMM_DEBUG = False


class LinS_AllGatherHandle:

    def __init__(self, handle, param: Parameter, quantization=None) -> None:
        if param.ds_status != ZeroParamStatus.INFLIGHT:
            raise RuntimeError(f"expected param {param.ds_summary()} to be available")

        self.__handle = handle
        self.__param = param
        self.__quantization = quantization

    def wait(self) -> None:
        instrument_w_nvtx(self.__handle.wait)()
        if self.__quantization:
            instrument_w_nvtx(self.__quantization.quant_handle.wait)()
            self.__param.data = self.__quantization.backend.dequantize(
                self.__quantization.quantized_param, self.__quantization.scale_buffer).to(self.__param.device)
        self.__param.ds_status = ZeroParamStatus.AVAILABLE
        # 恢复原有param的shape等元信息
        self.__param = self.__param.zero35_restore_allgahter_ds_tensor(self.__param)


class LinS_AllGatherCoalescedHandle(AllGatherCoalescedHandle):
    """ This handle assumes that no need to
    copy data out from a contiguous tensor
    """

    def __init__(
        self,
        allgather_handle,
        params: List[Parameter],
        partitions: List[Tensor],
        world_size: int,
        use_secondary_tensor=False,
        forward=False,
        quantization=None,
    ) -> None:
        self.allgather_handle = allgather_handle
        self.params = params
        self.partitions = partitions
        self.world_size = world_size
        self.use_secondary_tensor = use_secondary_tensor
        self.forward = forward
        self.complete = False
        self.quantization = quantization

        for param in self.params:
            if param.ds_status != ZeroParamStatus.INFLIGHT:
                raise RuntimeError(f"expected param {param.ds_summary()} to not be available")

    @instrument_w_nvtx
    def wait(self) -> None:
        if self.complete:
            return

        instrument_w_nvtx(self.allgather_handle.wait)()

        if self.quantization:
            instrument_w_nvtx(self.quantization.quant_handle.wait)()
            flat_tensor = self.quantization.backend.dequantize(
                self.quantization.quantized_param, self.quantization.scale_buffer).to(self.params[0].device)

            self.partitions: List[Parameter] = []
            for i in range(self.world_size):
                self.partitions.append(
                    flat_tensor.narrow(0, self.quantization.partition_sz * i, self.quantization.partition_sz))

        # split the single tensor out into individual tensors
        param_offset = 0
        for param in self.params:
            assert param.ds_status == ZeroParamStatus.INFLIGHT, f"expected param {param.ds_summary()} to be inflight"
            partitions: List[Tensor] = []
            ds_tensor_numel = param.ds_tensor.ds_numel
            if self.use_secondary_tensor and not self.forward:
                ds_tensor_numel *= param.ds_secondary_tensor_num_of_groups
            for rank in range(self.world_size):
                param_start = rank * ds_tensor_numel
                if param_start < param.ds_numel:
                    part_to_copy = self.partitions[rank].narrow(0, param_offset,
                                                                min(param.ds_numel - param_start, ds_tensor_numel))
                    partitions.append(part_to_copy)

            param.zero35_restore_allgahter_ds_tensor(param)
            param.data = instrument_w_nvtx(torch.cat)(partitions).view(param.ds_shape)
            param.ds_status = ZeroParamStatus.AVAILABLE

            for part_to_copy in partitions:
                if not get_accelerator().is_synchronized_device():
                    part_to_copy.record_stream(get_accelerator().current_stream())

            param_offset += ds_tensor_numel

        if self.now_mico_step_id() == 0 and self.forward:
            # 在每个micro_step 的第一次fwd时需要做全局的 all-gather
            # zero35_debug(f"now_mico_step_id:{self.now_mico_step_id()}, forward:{forward}, skip at count: {self.all_gahter_count} get: {param}!", flush=True)
            # TODO:  参数从 from os-partition to param-partition，只影响正确性，不影响性能测试
            pass
        else:
            # 在后续的 fwd/bwd 只需要做节点内的 all-gahter
            zero35_g_p_all_gather_coalesced([param])  # partition_type
            # zero35_debug(f"now_mico_step_id:{self.now_mico_step_id()}, forward:{forward}, do reshape finish at count: {self.all_gahter_count} get: {param}!", flush=True)

        self.complete = True


class LinS_Init(Init):

    def __init__(self,
                 module=None,
                 data_parallel_group=None,
                 mem_efficient_linear=True,
                 remote_device=None,
                 pin_memory=False,
                 config_dict_or_path=None,
                 config=None,
                 enabled=True,
                 dtype=None,
                 mpu=None):
        assert config_dict_or_path is not None, "Must provide configuration for MiCS Initialization"
        _ds_config = deepspeed.runtime.config.DeepSpeedConfig(config_dict_or_path, mpu)
        if not dist.is_initialized():
            dist.init_distributed()
            assert dist.is_initialized(), "Parameters cannot be scattered without initializing deepspeed.comm"

        self._dp_process_group = dist.get_world_group()
        self.hierarchical_allgather = _ds_config.zero_config.hierarchical_allgather
        self.world_size = dist.get_world_size(self._dp_process_group)
        # self.local_device = torch.device(get_accelerator().device_name(os.environ["LOCAL_RANK"]))

        if mpu is not None:
            self.model_parallel_group = mpu.get_model_parallel_group()
            self.model_parallel_rank = mpu.get_model_parallel_rank()
            self.model_parallel_size = dist.get_world_size(self.model_parallel_group)
            self.data_parallel_size = self.world_size // self.model_parallel_size
        else:
            self.model_parallel_size = 1
            self.model_parallel_group = None
            self.model_parallel_rank = 0
            self.data_parallel_size = self.world_size

        assert self.model_parallel_size == 1, "zero35现在不支持模型并行"

        global global_lins_utils
        if global_lins_utils is None:
            global_lins_utils = LinSProcessGroup(
                self._dp_process_group,
                lins_param_partition_num=_ds_config.zero_config.lins_param_partition_num,
                lins_grad_partition_num=_ds_config.zero_config.lins_grad_partition_num,
                lins_os_partition_num=_ds_config.zero_config.lins_os_partition_num)

        self._grad_process_group = global_lins_utils._grad_process_group
        self._param_process_group = global_lins_utils._param_process_group

        # hack掉 partition param 的方法
        self._allgather_param = self._allgather_param_lins
        self._allgather_params_coalesced = self._allgather_params_coalesced_lins

        super().__init__(module, data_parallel_group, mem_efficient_linear, remote_device, pin_memory,
                         config_dict_or_path, config, enabled, dtype, mpu)

    def _update_persist_config(self, ds_config):
        # persistence_threshold 我们以节点内切分大小为准
        with set_lins_parition_type(partition_type="param"):
            num_parition = self.num_partitions

        Init.apply_param_persistence = True
        Init.param_persistence_threshold = ds_config.zero_config.param_persistence_threshold
        Init.model_persistence_threshold = ds_config.zero_config.model_persistence_threshold // num_parition

    def _convert_to_deepspeed_param(self, param):
        super()._convert_to_deepspeed_param(param)
        print_rank_0("Lins: _convert_to_deepspeed_param", force=True)
        # attach communication groups to every param
        # param.comm = self.mics_comm_groups

        # record existing all_gather_coalesced implementation
        # so that we can fallback later
        old_all_gather_coalesced = param.all_gather_coalesced

        # def _param_all_gather_coalesced(params, param_buffers=None, **kwargs):
        #     """"""
        #     # mics_comm_groups: MiCS_CommGroups = params[0].comm
        #     # hierarchical_all_gather = has_hierarchical_all_gather_groups(mics_comm_groups)
        #     # if dist.has_coalescing_manager() and hierarchical_all_gather:
        #     #     return self._hierarchical_all_gather_params(params, param_buffers)
        #     # elif dist.has_coalescing_manager():
        #     #     return self._flat_all_gather_with_coalescing_manager(params, param_buffers)
        #     # else:
        #     #     return old_all_gather_coalesced(params, **kwargs)
        #     return self.lins_all_gather_coalesced(params, param_buffers=None, **kwargs)

        def partition(param_list=None, backward=False, hierarchy=0, has_been_updated=False):
            cls = param
            print_rank_0(f"{'--'*hierarchy}----Zero35 Partitioning param {debug_param2name_id_shape_device(cls)}",
                         force=False)
            if param_list is None:
                param_list = [cls]
            self._lins_partition(param_list, has_been_updated=has_been_updated)

        def padding_size(partition_type):
            return self._lins_padding_size(param, partition_type)

        def aligned_size(partition_type):
            return self._lins_aligned_size(param, partition_type)

        def partition_numel(partition_type=None):
            return self._lins_partition_numel(param, partition_type)

        def zero35_hack_allgahter_ds_tensor(mico_step, forward):

            see_memory_usage(f"before zero35_hack_allgahter_ds_tensor, mico_step:{mico_step}, forward:{forward}")
            if zero35_judge_gahter_boundary(mico_step, forward):
                # gather boundary
                partition_type = "os"

                assert hasattr(param, 'ds_numel'), 'zero35_hack_allgahter_ds_tensor input must be ds_param'
                parition_num = self.get_world_size(partition_type)

                node_id = dist.get_rank() // self.zero35_parallel_size
                partition_unit_size = param.ds_numel // parition_num
                param_ds_tensor = param.ds_tensor.view(-1, partition_unit_size)

                # backup ds_tensor
                param.ds_numel_backup = param.ds_tensor.ds_numel
                param.ds_tensor_backup = param.ds_tensor.data

                # hack ds_tensor
                param.ds_tensor = param_ds_tensor[node_id]
                assert param.ds_tensor.storage().data_ptr() == param_ds_tensor.storage().data_ptr()

                param.ds_tensor.ds_numel = partition_unit_size
                param.ds_tensor.status = PartitionedParamStatus.AVAILABLE
                param.ds_tensor.final_location = None
                param.ds_tensor.is_first_fwd_all_gahter = True

                # zero35_debug(f"zero35_hack_allgahter_ds_tensor DEBUG: mico_step: {mico_step}, forward:{forward}, param.ds_numel : {param.ds_numel}, get : {param_ds_tensor}, partition_type:{partition_type}, partition_unit_size:{partition_unit_size}", flush=True)
            else:
                partition_type = "param"
                partition_unit_size = param.ds_tensor.ds_numel
                # zero35_debug(f"zero35_hack_allgahter_ds_tensor DEBUG: mico_step: {mico_step}, forward:{forward}, SKIP hack, partition_unit_size:{partition_unit_size}", flush=True)

            see_memory_usage(f"after zero35_hack_allgahter_ds_tensor, mico_step:{mico_step}, forward:{forward}")
            return partition_unit_size

        def zero35_restore_allgahter_ds_tensor(param):
            # TODO:(wgt)
            if param.ds_tensor.is_first_fwd_all_gahter == True:
                # zero35_debug(f"now ds_tensor numel: {param.ds_tensor.ds_numel}, backup numel: {param.ds_numel_backup}")
                # zero35_debug(f"now ds_tensor data: {param.ds_tensor.data}, backup data: {param.ds_tensor_backup}")

                param.ds_tensor.data = param.ds_tensor_backup
                param.ds_tensor.is_first_fwd_all_gahter = False
                param.ds_tensor.ds_numel = param.ds_numel_backup

        # def get_partition_group(param):
        #     global
        #     return param.

        def lins_all_gahter_coalesced(params: Iterable[Parameter],
                                      forward: bool = True,
                                      safe_mode: bool = False,
                                      quantize: bool = False,
                                      mico_step: int = 0):
            if self.zero35_judge_gahter_boundary(mico_step, forward) and \
                self.hierarchical_allgather:
                return self.zero35_hierarchical_all_gather_params(params, forward, safe_mode, quantize, mico_step)
            else:
                return self.lins_all_gather_coalesced(params, forward, safe_mode, quantize, mico_step)

        # change the all_gather_coalesced method
        param.all_gather_coalesced = lins_all_gahter_coalesced
        param.partition = partition
        param.padding_size = padding_size
        param.aligned_size = aligned_size
        param.partition_numel = partition_numel
        # param.get_partition_group = get_partition_group

        param.zero35_hack_allgahter_ds_tensor = zero35_hack_allgahter_ds_tensor
        param.zero35_restore_allgahter_ds_tensor = zero35_restore_allgahter_ds_tensor

    def _lins_partition(self, param_list, force=False, has_been_updated=False):
        for param in param_list:
            print_rank_0(f"Before Zero35 Partitioning Param {param.ds_id}", force=False)
            self._lins_partition_param(param, has_been_updated=has_been_updated)
            param.ds_status = ZeroParamStatus.NOT_AVAILABLE

    def _lins_partition_numel(self, param, partition_type=None):
        if partition_type is not None:
            with set_lins_parition_type(partition_type=partition_type):
                tensor_size = self._lins_aligned_size(param, partition_type)
                return tensor_size // self.num_partitions
        else:
            tensor_size = self._lins_aligned_size(param, partition_type)
            return tensor_size // self.num_partitions

    def _lins_padding_size(self, param, partition_type=None):
        # param group 切分的 parition_unit 的大小要大于 os group
        # 如何返回正确的 padding ?
        if partition_type is not None:
            with set_lins_parition_type(partition_type):
                remainder = param.ds_numel % self.num_partitions
                return (self.num_partitions - remainder) if remainder else 0
        else:
            remainder = param.ds_numel % self.num_partitions
            return (self.num_partitions - remainder) if remainder else 0

    def _lins_aligned_size(self, param, partition_type=None):
        if partition_type is not None:
            with set_lins_parition_type(partition_type):
                return param.ds_numel + self._lins_padding_size(param, partition_type)
        else:
            return param.ds_numel + self._lins_padding_size(param, partition_type)

    def get_partition_dp_group(self, param):
        return param.ds_process_group

    def get_partition_rank(self):
        """subclass can overload to specify different relative rank in
        parameter partition group"""
        return dist.get_rank(self.get_process_group())

    @property
    def num_partitions(self):
        global global_parition_type
        _parition_type = global_parition_type['type']
        if _parition_type == "os":
            return dist.get_world_size(self._dp_process_group)
        elif _parition_type == "grad":
            return dist.get_world_size(self._grad_process_group)
        elif _parition_type == "param":
            return dist.get_world_size(self._param_process_group)
        else:
            assert False, f"unknown partition_type: {_parition_type}"

    def get_dp_process_group(self):
        """ Return the communication group with all data-parallel ranks """
        return self.ds_process_group

    def get_process_group(self):
        """ Return the communication group with all data-parallel ranks """
        global global_parition_type
        _parition_type = global_parition_type['type']
        if _parition_type == "os":
            return self._dp_process_group
        elif _parition_type == "grad":
            return self._grad_process_group
        elif _parition_type == "param":
            return self._param_process_group
        else:
            assert False, f"unknown partition_type: {_parition_type}"

    def get_param_process_group(self, param):
        """ Return the communication group with all data-parallel ranks """
        return param.get_param_process_group()

    """
    通信相关需要重载的方法，包括 partition param 和 stage3 的
    """

    def _allgather_param_lins(self, param, async_op=False, hierarchy=0):
        partition_size = param.ds_tensor.ds_numel

        with set_lins_parition_type(partition_type="param"):
            num_partitions = param.partition_numel
            aligned_param_size = param.aligned_size
            partition_all_gather_group = self.get_partition_group(param)

        tensor_size = partition_size * num_partitions
        assert tensor_size == aligned_param_size, f'param id {param.ds_id} aligned size {aligned_param_size} does not match tensor size {tensor_size}'
        print_rank_0(
            f"{'--'* hierarchy}---- Before allocating allgather param {debug_param2name_id_shape_status(param)} partition size={partition_size}"
        )

        see_memory_usage(
            f'Before allocate allgather param {debug_param2name_id_shape_status(param)} partition_size={partition_size} ',
            force=False)
        flat_tensor = torch.zeros(aligned_param_size, dtype=param.dtype, device=param.device).view(-1)
        see_memory_usage(
            f'After allocate allgather param {debug_param2name_id_shape_status(param)} {aligned_param_size} {partition_size} ',
            force=False)

        get_accelerator().synchronize()

        print_rank_0(
            f"{'--'* hierarchy}----allgather param with {debug_param2name_id_shape_status(param)} partition size={partition_size}"
        )
        handle = dist.all_gather_into_tensor(flat_tensor,
                                             param.ds_tensor.to(get_accelerator().device_name()),
                                             group=partition_all_gather_group,
                                             async_op=async_op)
        replicated_tensor = flat_tensor.narrow(0, 0, param.ds_numel).view(param.ds_shape)
        param.data = replicated_tensor.data
        return handle

    # all_gather_coalesced 和 _allgather_params_coalesced 是两个名字很像，但是干不同事情的 all-gather
    def lins_all_gather_coalesced(self,
                                  params: Iterable[Parameter],
                                  forward: bool = True,
                                  safe_mode: bool = False,
                                  quantize: bool = False,
                                  mico_step: int = 1) -> LinS_AllGatherCoalescedHandle:
        # fetches from nvme if the partition is not available and in nvme
        self._ensure_availability_of_partitioned_params(params)

        with set_lins_parition_type(partition_type="param"):
            num_partitions = self.num_partitions

        if num_partitions == 1:
            return _no_gather_coalesced(params)

        for param in params:
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(param.ds_summary())
            param.ds_status = ZeroParamStatus.INFLIGHT

        #use appropriate all gather process group
        partition_type = "os" if zero35_judge_gahter_boundary(mico_step, forward) else "param"
        with set_lins_parition_type(partition_type):
            ds_process_group = self.get_process_group()
            rank_in_group = self.get_partition_rank()
            world_size = self.num_partitions

        use_secondary_tensor = False

        params = sorted(params, key=lambda p: p.ds_id)

        if logger.isEnabledFor(logging.DEBUG):
            print_rank_0(f"-allgather_coalesced: {[p.ds_id for p in params]}")

        if safe_mode:
            assert_ints_same_as_other_ranks([p.ds_id for p in params])
            assert_ints_same_as_other_ranks([p.ds_tensor.ds_numel for p in params])

        if len(params) == 1:
            # have an opportunity to avoid some intermediate memory allocations
            param, = params
            self.zero35_hack_allgahter_ds_tensor(param, mico_step, forward)

            buffer_size = math.ceil(param.ds_numel / world_size) * world_size
            param_ds_tensor = param.ds_tensor
            param_buffer = torch.empty(
                buffer_size,
                dtype=param_ds_tensor.dtype if not quantize else torch.int8,
                device=get_accelerator().current_device_name(),
                requires_grad=False,
            )
            handles = _dist_allgather_fn(
                param_ds_tensor.to(get_accelerator().current_device_name()),
                param_buffer,
                ds_process_group,
            )
            param.data = param_buffer.narrow(0, 0, param.ds_numel).view(param.ds_shape).to(param.device)
            return AllGatherHandle(handles, param)
        else:
            partition_sz = 0
            for param in params:
                partition_sz += self.zero35_hack_allgahter_ds_tensor(param, mico_step, forward)

            flat_tensor = torch.empty(partition_sz * world_size,
                                      dtype=get_only_unique_item(p.ds_tensor.dtype
                                                                 for p in params) if not quantize else torch.int8,
                                      device=get_accelerator().current_device_name(),
                                      requires_grad=False)
            partitions: List[Parameter] = []
            for i in range(world_size):
                partitions.append(flat_tensor.narrow(0, partition_sz * i, partition_sz))
            instrument_w_nvtx(torch.cat)([p.ds_tensor.to(get_accelerator().current_device_name()) for p in params],
                                         out=partitions[rank_in_group])
            handle = _dist_allgather_fn(partitions[rank_in_group], flat_tensor, ds_process_group)

            return LinS_AllGatherCoalescedHandle(
                allgather_handle=handle,
                params=params,
                partitions=partitions,
                world_size=world_size,
                use_secondary_tensor=use_secondary_tensor,
                forward=forward,
            )

    def _allgather_params_coalesced_lins(self, param_list, hierarchy=0, quantize=False):
        """ blocking call
        avoid explicit memory copy in _allgather_params
        """
        assert False

    @instrument_w_nvtx
    def _lins_partition_param(self, param, buffer=None, has_been_updated=False):
        """
        zero35 进行 partition的基本单元是按照 dp 范围切分，这些切分部分我们称之为 'partition_unit'
        每个rank可能会分到多个 'partition_unit'

        Args:
            param (_type_): _description_
            buffer (_type_, optional): _description_. Defaults to None.
            has_been_updated (bool, optional): _description_. Defaults to False.
        """
        assert param.ds_status is not ZeroParamStatus.INFLIGHT, f" {param} Cannot partition a param in flight"
        global reuse_buffers
        print_rank_0(f"Param id {param.ds_id} status is {param.ds_status}", force=False)
        # # zero35_debug(f"do _lins_partition_param!")

        param_comm_group = self._param_process_group
        dp_comm_group = self._dp_process_group
        param_num_partitions = dist.get_world_size(param_comm_group)
        dp_num_partitions = dist.get_world_size(dp_comm_group)

        # assert param_num_partitions == 8, "zero35 split param in local node device"
        # if param_num_partitions != 8:
        #     print(f"zero35 split param worldisze: {param_num_partitions}", flush=True)

        if param.ds_status is ZeroParamStatus.AVAILABLE:
            print_rank_0(f"Partitioning param id {param.ds_id} reuse buffers {reuse_buffers}", force=False)

            if param.ds_tensor is not None and not has_been_updated:  ##param already partitioned
                see_memory_usage(f'Before partitioning param 2:{param.ds_id} {param.shape}', force=False)
                # param.data does not store anything meaningful in partitioned state
                free_param(param)
                see_memory_usage(f'After partitioning param 2:{param.ds_id} {param.shape}', force=False)
                return

            tensor_size_param = self._lins_aligned_size(param, partition_type="param")
            tensor_size_dp = self._lins_aligned_size(param, partition_type="os")

            # assert tensor_size_dp == tensor_size_param, f"different padding size: tensor_size_dp:{tensor_size_dp} = tensor_size_param: {tensor_size_param}"
            tensor_size = tensor_size_dp

            partition_size = tensor_size // param_num_partitions
            unit_partition_size = tensor_size // dp_num_partitions

            assert partition_size % unit_partition_size == 0

            if param.ds_tensor is None:
                final_location = None
                # assert param.ds_persist is False, "ds_persist可能会有ug"
                if param.ds_persist:
                    device = self.local_device
                else:
                    device = self.remote_device

                # buffer 大小仍然是 partition_size 这么大
                partitioned_tensor = torch.empty(partition_size, dtype=param.dtype, device=device)
                partitioned_tensor.requires_grad = False
                param.ds_tensor = partitioned_tensor  # 被切分后的tensor
                param.ds_tensor.ds_numel = partition_size  # ds_numel 是 buffer大小，等于partition_size，但每个param实际上仍然是被切分了 1/dp 份
                param.unit_partition_size = unit_partition_size  # 这里存一份 unit_partition_size，给 _unflatten_partitioned_parameters 用
                param.ds_tensor.is_first_fwd_all_gahter = False

                param.ds_tensor.status = PartitionedParamStatus.AVAILABLE
                param.ds_tensor.final_location = final_location

            partition_unit_num = partition_size // unit_partition_size

            assert tensor_size % partition_unit_num == 0
            partition_stride = tensor_size // partition_unit_num

            # 这里需要改成分段拷贝
            try:
                with set_lins_parition_type(partition_type="param"):
                    print(
                        f"partition type: {global_parition_type}, self.get_partition_rank():{self.get_partition_rank()}",
                        flush=True)
                    offset = unit_partition_size * self.get_partition_rank()
                    one_dim_param = param.contiguous().view(-1)

                    # # zero35_debug(f"Rank: {os.environ['SLURM_PROCID']}, partition_unit_num: {partition_unit_num}, partition_stride:{partition_stride}, offset:{offset}, ", flush=True)
                    for pdx in range(partition_unit_num):
                        start = offset + partition_stride * pdx
                        sub_start = unit_partition_size * pdx
                        # # zero35_debug(f"Rank: {os.environ['SLURM_PROCID']}, pdx:{pdx},start: {start}, sub_start:{sub_start}" , flush=True)

                        if self.get_partition_rank() == param_num_partitions -1 \
                            and pdx == partition_unit_num - 1:   # 只有最后一块 1/dp 的 partition unit 需要补齐 padding
                            param.ds_tensor[sub_start:] = one_dim_param[start:]
                        else:
                            param.ds_tensor[sub_start:sub_start +
                                            unit_partition_size] = one_dim_param[start:start + unit_partition_size]
            except Exception as e:
                print(f"catch exception: {e}", flush=True)
                import pdb
                pdb.set_trace()

            # if os.environ['SLURM_PROCID'] == '0':
            # # zero35_debug(f"Rank: {os.environ['SLURM_PROCID']}, partition param done {param.ds_tensor}", flush=True)

            see_memory_usage(f'Before partitioning param {param.ds_id} {param.shape}', force=ENBALE_MEM_DEBUG)
            # # zero35_debug(f"Before partitioning param ID {param.ds_id} partitioned type {param.dtype} dev {param.device} shape {param.shape}")
            free_param(param)
            # # zero35_debug(f"After partitioning param ID {param.ds_id} partitioned type {param.dtype} dev {param.device} shape {param.shape}")
            see_memory_usage(f'After partitioning param {param.ds_id} {param.shape}', force=ENBALE_MEM_DEBUG)

    @instrument_w_nvtx
    def zero35_hierarchical_all_gather_params(self,
                                              params: Iterable[Parameter],
                                              forward: bool = True,
                                              safe_mode: bool = False,
                                              quantize: bool = False,
                                              mico_step: int = 1):

        params_buffers = None

        # self._ensure_availability_of_partitioned_params(params)
        from deepspeed.runtime.zero.mics import MiCS_AllGatherCoalescedHandle
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

        for param in params:
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(param.ds_summary())
            param.ds_status = ZeroParamStatus.INFLIGHT

        # ensure that each rank has params in same order. the allgather
        # is done by flattening the parameter list into a single tensor that
        # can be allgathered in a single call - this means that if each rank
        # gives a list of the same parameters in a different order we will
        # silently get incorrect parameter values, and have very difficult
        # to debug correctness issues.
        params = sorted(params, key=lambda p: p.ds_id)

        local_rank = dist.get_rank(group=self.zero35_group)
        inter_node_comm_group = self.zero35_hierarchical_group
        intra_node_comm_group = self.zero35_group
        intra_param_shard_size = dist.get_world_size(intra_node_comm_group)
        assert intra_param_shard_size == 8

        dp_param_shard_size = dist.get_world_size()

        inter_node_size = dist.get_world_size(group=inter_node_comm_group)
        intra_node_size = dist.get_world_size(group=intra_node_comm_group)
        param_tensors = []
        for i, p in enumerate(params):
            self.zero35_hack_allgahter_ds_tensor(p, mico_step, forward)
            param_size = p.ds_tensor.ds_numel * dp_param_shard_size  # unit_size * dp world size
            #zero35_debug(F"param_size :{param_size / dp_param_shard_size}, intra_param_shard_size: {intra_param_shard_size}, dp_param_shard_size, :{dp_param_shard_size/dp_param_shard_size}, p.ds_tensor.ds_numel:{p.ds_tensor.ds_numel/dp_param_shard_size}, p.ds_tensor.ds_shape:{p.ds_shape}", force=True)
            if params_buffers is not None and params_buffers[i] is not None:
                assert params_buffers[i].numel(
                ) == param_size, f'param_buffers[{i}] size {params_buffers[i].numel()} does not match with param_size {param_size}'
                param_tensor = params_buffers[i]
            else:
                param_tensor = torch.empty(param_size, dtype=p.dtype, device=self.local_device,
                                           requires_grad=False).view(-1)
            param_tensors.append(param_tensor)

        # inter node all-gather
        inter_outputs = []
        inter_inputs = []
        try:
            for i, p in enumerate(params):
                inter_size = p.ds_tensor.ds_numel * inter_node_size
                #zero35_debug(f"p.ds_tensor.ds_numel:{p.ds_tensor.ds_numel/dp_param_shard_size}, inter_node_size:{inter_node_size}, inter_size: {inter_size/dp_param_shard_size}, p.ds_tensor.ds_shape:{p.ds_shape}", force=True)
                _out = param_tensors[i].narrow(0, local_rank * inter_size, inter_size)
                inter_outputs.append(_out)
                inter_inputs.append(p.ds_tensor.data.view(-1).to(self.local_device))
                #zero35_debug(f"inter input unit parma size: {p.ds_tensor.data.numel()/dp_param_shard_size}, inter_size: {inter_size/dp_param_shard_size}", force=True)
        except Exception as e:
            import time
            if dist.get_rank() == 0:
                print(e, flush=True)
            time.sleep(10000)

        # sync enqueue
        dist.all_gather_coalesced(inter_outputs, inter_inputs, group=inter_node_comm_group, async_op=False)

        # intra node all-gather
        intra_outputs = []
        intra_inputs = []
        for i, p in enumerate(params):
            # partition param into multiple chunks for allgather
            # because inter-node all-gather outputs are in a continues memory
            # while in param memory, those inter-node data are placed in different
            # location.
            # each chunk is an intra-node output
            try:
                #zero35_debug(f"inter_node_size:{inter_node_size}, intra_node_size:{intra_node_size}, p.ds_tensor.ds_numel:{p.ds_tensor.ds_numel}, local_rank:{local_rank}", force=True)
                param_chunk = param_tensors[i].view(
                    (inter_node_size, intra_node_size, p.ds_tensor.ds_numel)).narrow(1, local_rank, 1)
                param_chunk.copy_(inter_outputs[i].detach().clone().view(param_chunk.size()))
                output_chunks = torch.chunk(param_tensors[i], inter_node_size)
                for j, _out in enumerate(output_chunks):
                    intra_chunk_size = intra_node_size * p.ds_tensor.ds_numel
                    local_offset = local_rank * p.ds_tensor.ds_numel
                    #zero35_debug(f"intra input unit parma size: {p.ds_tensor.data.numel()/dp_param_shard_size}, intra_chunk_size: {intra_chunk_size/dp_param_shard_size}", force=True)
                    _in = param_tensors[i].narrow(0, j * intra_chunk_size + local_offset, p.ds_tensor.ds_numel)
                    intra_outputs.append(_out)
                    intra_inputs.append(_in)
            except Exception as e:
                import time
                if dist.get_rank() == 0:
                    print(e, flush=True)
                time.sleep(10000)

        all_gather_handle = dist.all_gather_coalesced(intra_outputs,
                                                      intra_inputs,
                                                      group=intra_node_comm_group,
                                                      async_op=True)
        for i, param in enumerate(params):
            param.data = param_tensors[i].narrow(0, 0, param.ds_numel).view(param.ds_shape).data
            #zero35_debug(f"finish pre allgather param.data:{param.data.numel()/dp_param_shard_size}, ds_tensor.numel: {param.ds_tensor.numel()/dp_param_shard_size}", force=True)

        # import time
        # time.sleep(10000)
        return MiCS_AllGatherCoalescedHandle(
            allgather_handle=all_gather_handle,
            params=params,
            partitions=[],
            world_size=intra_param_shard_size,
        )


class LinS_Optimizer(DeepSpeedZeroOptimizer_Stage3):
    """
    MiCS Optimizer
    """

    def __init__(self,
                 module,
                 init_optimizer,
                 timers,
                 ds_config,
                 static_loss_scale=1,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True,
                 contiguous_gradients=True,
                 reduce_bucket_size=500000000,
                 prefetch_bucket_size=50000000,
                 max_reuse_distance=1000000000,
                 max_live_parameters=1000000000,
                 param_persistence_threshold=100000,
                 model_persistence_threshold=sys.maxsize,
                 dp_process_group=None,
                 reduce_scatter=True,
                 overlap_comm=False,
                 offload_optimizer_config=None,
                 offload_param_config=None,
                 sub_group_size=1000000000000,
                 mpu=None,
                 clip_grad=0,
                 gradient_accumulation_dtype=torch.float16,
                 communication_data_type=torch.float16,
                 postscale_gradients=True,
                 gradient_predivide_factor=1,
                 gradient_accumulation_steps=1,
                 elastic_checkpoint=False,
                 aio_config=None):

        log_dist("Init LinS optimizer", ranks=[0])

        self._param_partition_unit_size = []

        global global_lins_utils
        self._grad_process_group = global_lins_utils._grad_process_group
        self._param_process_group = global_lins_utils._param_process_group

        super().__init__(module, init_optimizer, timers, ds_config, static_loss_scale, dynamic_loss_scale,
                         dynamic_loss_args, verbose, contiguous_gradients, reduce_bucket_size, prefetch_bucket_size,
                         max_reuse_distance, max_live_parameters, param_persistence_threshold,
                         model_persistence_threshold, dp_process_group, reduce_scatter, overlap_comm,
                         offload_optimizer_config, offload_param_config, sub_group_size, mpu, clip_grad,
                         gradient_accumulation_dtype, communication_data_type, postscale_gradients,
                         gradient_predivide_factor, gradient_accumulation_steps, elastic_checkpoint, aio_config)
        first_param = next(module.parameters())
        # overload the dp_process_group and partition_count
        # self.dp_process_group = first_param.comm.param_shard_group
        # self.partition_count = first_param.comm.param_shard_size

    def cal_tensor_size(self, param):
        ds_t_size = 0
        dtype = param.dtype
        if dtype == torch.float32:
            unit_byte = 4
        elif dtype == torch.float16 or dtype == torch.bfloat16:
            unit_byte = 2
        else:
            assert False, f"Unexpected dtype: {dtype}!"

        if hasattr(param, 'ds_tensor'):
            ds_t_size = param.ds_tensor.numel()
            torch_t_size = param.numel()
            assert ds_t_size >= torch_t_size, f"{ds_t_size} >= {torch_t_size}"
            assert param.ds_tensor.dtype == param.dtype
            if param.grad is not None:
                assert not hasattr(param.grad, 'ds_tensor')
                assert param.grad.numel() == 0
        else:
            ds_t_size = param.numel()

        return ds_t_size * unit_byte

    def format_size(self, size):
        if size < 1024:
            return f"{size} B"
        elif size >= 1024 and size < 1024**2:
            return f"{size / 1024:.2f} KB"
        elif size >= 1024**2 and size < 1024**3:
            return f"{size / 1024**2:.2f} MB"
        else:
            return f"{size / 1024**3:.2f} GB"

    def cal_stage3_mem_usage(self, all_params):
        grad_partitions_flat_buffer_size = self.cal_tensor_size(self.grad_partitions_flat_buffer)
        ipg_bucket_flat_buffer_size = self.cal_tensor_size(self.__ipg_bucket_flat_buffer)

        fp32_partitioned_groups_flat_size = 0
        for i in range(len(self.fp32_partitioned_groups_flat)):
            partitioned_groups_flat = self.fp32_partitioned_groups_flat[i]
            fp32_partitioned_groups_flat_size += self.cal_tensor_size(partitioned_groups_flat)

        # zero35_debug(f"Grad size: {self.format_size(grad_partitions_flat_buffer_size)}", force=True)
        # zero35_debug(f"fp32 os size: {self.format_size(fp32_partitioned_groups_flat_size)}", force=True)
        # zero35_debug(f"os size total: {self.format_size(fp32_partitioned_groups_flat_size * 3)}", force=True)
        # zero35_debug(f"ipg_bucket_flat_buffer_size: {self.format_size(ipg_bucket_flat_buffer_size)}", force=True)

        param_size = 0
        os_size = 0
        for param in all_params:
            param_size += self.cal_tensor_size(param)

        # zero35_debug(f"param_size: {self.format_size(param_size)}", force=True)
        # zero35_debug(f"all size: {self.format_size(param_size + grad_partitions_flat_buffer_size + ipg_bucket_flat_buffer_size + fp32_partitioned_groups_flat_size)}", force=True)

    """
    切分相关需要重载的函数
    """

    def bookkeeping_param_group(self, param_groups):
        print("bookkeeping_param_group!!!", flush=True)
        with set_lins_parition_type(partition_type="os"):
            for param_group_idx, param_group in enumerate(param_groups):
                for sub_group in param_group:
                    sub_group_idx = len(self.fp16_groups)

                    # record sub group and partitions
                    self.fp16_groups.append(sub_group)
                    self.fp16_partitioned_groups.append([param.ds_tensor for param in sub_group])

                    # record total elements of parameter partitions in sub group
                    assert self._param_partition_unit_size is not None
                    self._param_partition_unit_size.append([param.unit_partition_size for param in sub_group])
                    self.fp16_partitioned_groups_flat_numel.append(sum(p.partition_numel() for p in sub_group))
                    param_process_group = self._param_process_group

                    # record sub group -> group mapping
                    self.sub_group_to_group_id[sub_group_idx] = param_group_idx

                    # record padding required to align group to world size (only applies to last rank)
                    rank_requires_padding = dist.get_rank(
                        param_process_group) == dist.get_world_size(param_process_group) - 1
                    self.groups_padding.append(
                        [p.padding_size(partition_type="os") if rank_requires_padding else 0 for p in sub_group])

    def get_sub_p_g_parition(self, ds_param, grad=None):
        assert hasattr(ds_param, 'ds_numel'), 'get_sub_p_g_parition input must be ds_param'
        zero35_rank = dist.get_rank() // self.zero35_parallel_size  # TODO, remove hard code
        partition_unit_size = self.get_partition_unit_size(ds_param.ds_numel)
        if grad is not None:
            reshape_grad = grad.view(-1, partition_unit_size)[zero35_rank]
            assert reshape_grad.storage().data_ptr() == grad.storage().data_ptr()
            return reshape_grad
        else:
            reshape_param = ds_param.ds_tensor.view(-1, partition_unit_size)[zero35_rank]
            assert reshape_param.storage().data_ptr() == ds_param.ds_tensor.storage().data_ptr()
            return reshape_param

    @instrument_w_nvtx
    def independent_gradient_partition_epilogue(self):
        self.report_ipg_memory_usage(f"In ipg_epilogue before reduce_ipg_grads", 0)
        self.__reduce_and_partition_ipg_grads()
        self.report_ipg_memory_usage(f"In ipg_epilogue after reduce_ipg_grads", 0)

        if not get_accelerator().is_synchronized_device():
            self.reduce_and_partition_stream.synchronize()

        #in case of cpu offload, averaged gradients are already in fp32_partitioned_groups_flat.grad
        #TODO: use a similar code path for both cpu_offload and non-cpu offload
        if not self.offload_optimizer:
            for i, sub_group in enumerate(self.fp16_groups):
                #TODO: This is redundant
                self.averaged_gradients[i] = [
                    self.get_sub_p_g_parition(param, self.__param_id_to_grad_partition[param.ds_id])
                    if param.requires_grad else torch.zeros_like(param.ds_tensor) for param in sub_group
                ]
        # this method gets called after every backward. need to increment
        # here because if it gets incremented in backward() the micro step
        # id will be off by one when we do the reduce and partition at the.
        # start of this method.
        # TODO. make this less error prone
        self.micro_step_id += 1
        self._get_param_coordinator(training=True).micro_step_id += 1

    def _setup_for_real_optimizer(self):
        see_memory_usage("Before creating fp32 partitions", force=True)
        self._create_fp32_partitions()
        see_memory_usage("After creating fp32 partitions", force=True)
        dist.barrier()

        # To support pipelined optimizer swapping
        self._create_next_swappable_fp32_groups()

        see_memory_usage("Before initializing optimizer states", force=True)

        self.initialize_optimizer_states()
        see_memory_usage("After initializing optimizer states", force=True)
        dist.barrier()

        if dist.get_rank() == 0:
            logger.info(f"optimizer state initialized")

        # IPG
        if self.contiguous_gradients:
            self.__ipg_bucket_flat_buffer: Tensor = torch.empty(self.reduce_bucket_size,
                                                                dtype=self.dtype,
                                                                device=get_accelerator().current_device_name())

        self.grad_partitions_flat_buffer = None
        self.__param_id_to_grad_partition: Dict[int, Tensor] = {}

        all_params = list(itertools.chain.from_iterable(self.fp16_groups))

        with set_lins_parition_type(partition_type="grad"):
            self.grad_partitions_flat_buffer: Tensor = torch.zeros(sum(p.partition_numel() for p in all_params),
                                                                   dtype=self.gradient_accumulation_dtype,
                                                                   device=self.device)
            if self.offload_optimizer_pin_memory:
                self.grad_partitions_flat_buffer = get_accelerator().pin_memory(self.grad_partitions_flat_buffer)

            offset = 0
            for param in all_params:
                # self.__param_id_to_grad_partition[param.ds_id] = self.grad_partitions_flat_buffer.narrow(
                #     0, offset, param.partition_numel())
                # offset += param.partition_numel()
                self.__param_id_to_grad_partition[param.ds_id] = \
                    self.grad_partitions_flat_buffer.narrow(0, offset, param.partition_numel())
                offset += param.partition_numel()

        see_memory_usage("End of _setup_for_real_optimizer", force=True)
        return all_params

    def _create_fp16_sub_groups(self, params_group):
        params_group_numel = sum([param.partition_numel(partition_type="os") for param in params_group])
        sub_group_size = self.sub_group_size

        if sub_group_size is None or sub_group_size >= params_group_numel:
            return [params_group]

        sub_groups = []
        sub_group = []
        local_sub_group_size = 0
        for param in params_group:

            sub_group.append(param)
            local_sub_group_size += param.partition_numel(partition_type="os")

            if local_sub_group_size >= sub_group_size or id(param) == id(params_group[-1]):

                sub_groups.append(sub_group)

                sub_group = []
                local_sub_group_size = 0

        return sub_groups

    """
    通信相关需要重载的函数
    """

    @instrument_w_nvtx
    def __avg_scatter_grads(self, params_to_reduce: List[Parameter]) -> List[Tensor]:
        """average gradients and scatter partitions across ranks"""
        see_memory_usage(f"before __avg_scatter_grads, boundary: {self.zero35_judge_grad_boundary()}",
                         force=ENBALE_MEM_DEBUG)

        full_grads_for_rank = [p.grad for p in params_to_reduce]
        if self.communication_data_type != self.dtype:
            full_grads_for_rank = [g.to(self.communication_data_type) for g in full_grads_for_rank]

        if self.postscale_gradients and self.gradient_predivide_factor != 1.0:
            full_grads_for_rank = [g.div(self.gradient_predivide_factor) for g in full_grads_for_rank]

        if not self.zero35_judge_grad_boundary():
            # 梯度累加
            # 假设 dp_world_size = 8, 每个节点只有4张卡
            # os 切分:
            #       节点1: [[0], [1], [2], [3]], 节点2: [[4], [5], [6], [7]]
            # grad/param切分:
            #       节点1：[[0, 4], [1, 5] ,[2, 6], [3, 7]], 节点2: [[0, 4], [1, 5] ,[2, 6], [3, 7]]
            # [[0, 1, 2, 3], [4, 5, 6, 7]] -> [[0], 1, 2, 3,  [16], 5, 6, 7]
            # [[0, 1, 2, 3], [4, 5, 6, 7]] -> [0, [4], 2, 3,  4, [20], 6, 7]
            # [[0, 1, 2, 3], [4, 5, 6, 7]] -> [0, 1, [8], 3,  4, 5, [24], 7]
            # [[0, 1, 2, 3], [4, 5, 6, 7]] -> [0, 1, 2, [12], 4, 5, 6, [28]]]
            full_grads_for_rank, scatter_comm_group = zero35_g_p_reduce_scatter_coalesced(full_grads_for_rank,
                                                                                          partition_type="grad")

            # zero35_debug(f"Rank: {os.environ['SLURM_PROCID']}, mico_step: {self.micro_step_id}, before __avg_scatter_grads, after grad_tensor_list : {full_grads_for_rank}", flush=True)    #     before grad_tensor_list: {tensor_list_debug},  \
        else:
            # boundary
            # 在 boundary 阶段，进行 dp 范围的 reduce-scatter，
            # [0, 1, 2, 3, 4, 5, 6, 7]  -> [[0], 1, 2, 3, 4, 5, 6, 7]
            # [0, 1, 2, 3, 4, 5, 6, 7]  -> [0, [8], 2, 3, 4, 5, 6, 7]
            # [0, 1, 2, 3, 4, 5, 6, 7]  -> [0, 1, [16], 3, 4, 5, 6, 7]
            # [0, 1, 2, 3, 4, 5, 6, 7]  -> [0, 1, 2, [24], 4, 5, 6, 7]
            # [0, 1, 2, 3, 4, 5, 6, 7]  -> [0, 1, 2, 3, [32], 5, 6, 7]
            # [0, 1, 2, 3, 4, 5, 6, 7]  -> [0, 1, 2, 3, 4, [40], 6, 7]
            # [0, 1, 2, 3, 4, 5, 6, 7]  -> [0, 1, 2, 3, 4, 5, [48], 7]
            # [0, 1, 2, 3, 4, 5, 6, 7]  -> [0, 1, 2, 3, 4, 5, 6, [56]]
            # zero35_debug(f"Rank: {os.environ['SLURM_PROCID']}, mico_step: {self.micro_step_id}, skip zero35_g_p_reduce_scatter_coalesced, before __avg_scatter_grads, before grad_tensor_list", flush=True)
            scatter_comm_group = self.get_dp_process_group(partition_type="os")

        local_world_size = get_accelerator().device_count()
        global_world_size = dist.get_world_size()
        num_nodes = global_world_size // local_world_size

        if ENBALE_COMM_DEBUG:
            if dist.get_rank() == 0:
                numels = sum([p.numel() for p in full_grads_for_rank])
                print(f"__avg_scatter_grads, \
comm_group:{dist.get_world_size(scatter_comm_group)} \
nums:{numels}, \
size: {numels* full_grads_for_rank[0].element_size()/ (1024**2):.4f} MB",
                      flush=True)

        # # zero35_debug(f"before __avg_scatter_grads: {full_grads_for_rank}", flush=True)
        grad_partitions_for_rank = reduce_scatter_coalesced(full_grads_for_rank, scatter_comm_group)
        # # zero35_debug(f"after __avg_scatter_grads: {full_grads_for_rank}", flush=True)

        if self.postscale_gradients and self.gradient_predivide_factor != 1.0 and self.gradient_predivide_factor != dist.get_world_size(
                self.dp_process_group):
            grad_partitions_for_rank = [g.mul(self.gradient_predivide_factor) for g in grad_partitions_for_rank]

        if self.communication_data_type != self.dtype:
            grad_partitions_for_rank = [g.to(self.dtype) for g in grad_partitions_for_rank]

        see_memory_usage(f"after __avg_scatter_grads, boundary: {self.zero35_judge_grad_boundary()}",
                         force=ENBALE_MEM_DEBUG)

        return grad_partitions_for_rank

    @instrument_w_nvtx
    def __avg_scatter_grads_hierarchical(self, params_to_reduce: List[Parameter]) -> List[Tensor]:
        see_memory_usage(f"before __avg_scatter_grads, boundary: {self.zero35_judge_grad_boundary()}",
                         force=ENBALE_MEM_DEBUG)
        full_grads_for_rank = [p.grad for p in params_to_reduce]

        if self.communication_data_type != self.dtype:
            full_grads_for_rank = [g.to(self.communication_data_type) for g in full_grads_for_rank]

        if self.postscale_gradients and self.gradient_predivide_factor != 1.0:
            full_grads_for_rank = [g.div(self.gradient_predivide_factor) for g in full_grads_for_rank]

        inter_comm_group = self.zero35_hierarchical_group
        intra_comm_group = self.zero35_group

        assert dist.get_world_size(intra_comm_group) == 8

        def count_list_params_numel(grads_list):
            all_numel = 0
            for grad in grads_list:
                all_numel += grad.numel()
            return all_numel

        if not self.zero35_judge_grad_boundary():
            # full_grads_for_rank, scatter_comm_group = zero35_g_p_reduce_scatter_coalesced(full_grads_for_rank, partition_type="grad")
            # Replace
            grad_partitions_for_rank = reduce_scatter_coalesced(full_grads_for_rank, intra_comm_group)
        else:
            # boundary
            # inter reduce-scatter
            zero35_debug(f"before first educe_scatter: {count_list_params_numel(full_grads_for_rank)}", force=False)
            grad_partitions_for_rank = reduce_scatter_coalesced(full_grads_for_rank, inter_comm_group)
            zero35_debug(f"after first educe_scatter: {count_list_params_numel(grad_partitions_for_rank)}",
                         force=False)

            # Replace
            # intra reduce-scatter
            grad_partitions_for_rank = reduce_scatter_coalesced(grad_partitions_for_rank, intra_comm_group)
            zero35_debug(f"after sec educe_scatter: {count_list_params_numel(grad_partitions_for_rank)}", force=False)

        if self.postscale_gradients and self.gradient_predivide_factor != 1.0 and self.gradient_predivide_factor != dist.get_world_size(
                self.dp_process_group):
            grad_partitions_for_rank = [g.mul(self.gradient_predivide_factor) for g in grad_partitions_for_rank]

        if self.communication_data_type != self.dtype:
            grad_partitions_for_rank = [g.to(self.dtype) for g in grad_partitions_for_rank]

        see_memory_usage(f"after __avg_scatter_grads, boundary: {self.zero35_judge_grad_boundary()}",
                         force=ENBALE_MEM_DEBUG)

        return grad_partitions_for_rank

    def _unflatten_partitioned_parameters(self, sub_group_id):

        def get_sub_p_g_parition_from_torch_tensor(torch_tensor, i):
            zero35_rank = dist.get_rank() // self.zero35_parallel_size  # TODO, remove hard code
            # zero35_debug(f"self._param_partition_unit_size[sub_group_id]: {self._param_partition_unit_size[sub_group_id]}", flush=ENBALE_MEM_DEBUG)
            return torch_tensor.reshape(-1, self._param_partition_unit_size[sub_group_id][i])[zero35_rank]

        sub_fp16_partitioned_groups_flat = self.fp16_partitioned_groups_flat[sub_group_id]
        sub_fp16_partitioned_groups = [
            get_sub_p_g_parition_from_torch_tensor(param, i)
            for i, param in enumerate(self.fp16_partitioned_groups[sub_group_id])
        ]

        updated_params = self.unflatten(sub_fp16_partitioned_groups_flat, sub_fp16_partitioned_groups)

        # (TODO):wgt 为了简单，这里直接注释掉了，为了防止 intra 划分的 ds_tensor 覆盖 dp 划分的 ds_tensor
        # for partitioned_param, q in zip(self.fp16_partitioned_groups[sub_group_id], updated_params):
        #     partitioned_param.data = q.data

    @instrument_w_nvtx
    def partition_grads(self, params_to_release: List[Parameter], grad_partitions: List[Tensor]) -> None:
        offload_fp32_gradients = {}
        offload_fp32_offsets = {}
        buffers = []
        see_memory_usage(f"before partition_grads", force=ENBALE_MEM_DEBUG)
        for param, grad_partition in zip(params_to_release, grad_partitions):

            if self.micro_step_id == self.gradient_accumulation_steps - 1:  # bounary
                grad_rank = dist.get_rank(self.get_dp_process_group(partition_type="os"))
            else:
                grad_rank = dist.get_rank(self.get_dp_process_group(partition_type="grad"))

            with set_lins_parition_type(partition_type="grad"):
                contains_real_data = param.partition_numel() * grad_rank < param.ds_numel

            if not contains_real_data:
                # this grad partition is empty - don't need to do anything
                param.grad = None
                continue

            # move or accumulate gradient partition to target buffer
            grad_buffer = self.__param_id_to_grad_partition[param.ds_id].narrow(0, 0, grad_partition.numel())
            buffers.append(grad_buffer)
            if self.micro_step_id == 0:  # don't accumulate
                grad_buffer.copy_(grad_partition, non_blocking=True)
                # ensure grad buffer is a CUDA buffer to speed up the next few
                # operations and so it can be used asynchronously
                grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
            elif get_accelerator().on_accelerator(grad_buffer):
                grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype).view(grad_buffer.shape))
            else:
                # if dst is CPU, copy first to src device, do the addition
                # there, then move back to dst. adding directly to cpu is very slow
                cuda_grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
                cuda_grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype).view(cuda_grad_buffer.shape))
                grad_buffer.copy_(cuda_grad_buffer, non_blocking=True)
                # ensure grad buffer is a CUDA buffer to speed up the next few
                # operations and so it can be used asynchronously
                grad_buffer = cuda_grad_buffer

            # offload the gradient partition if applicable
            if self.offload_optimizer:
                i, dest_offset, _ = self.grad_position[self.get_param_id(param)]
                offload_fp32_gradients = {}
                offload_fp32_offsets = {}

                if self.is_gradient_accumulation_boundary:
                    self.norm_for_param_grads[self.get_param_id(param)] = self._constant_buffered_norm2(grad_buffer)

                    if self._swappable_optimizer_subgroup(i):
                        if not i in offload_fp32_gradients.keys():
                            offload_fp32_gradients[i] = []
                            offload_fp32_offsets[i] = []

                        offload_fp32_gradients[i].append(grad_buffer.float())
                        offload_fp32_offsets[i].append(dest_offset)
                    else:
                        fp32_grad_tensor = self.fp32_partitioned_groups_flat[i].grad.narrow(
                            0, dest_offset, grad_buffer.numel())
                        fp32_grad_tensor.copy_(grad_buffer)

            # free the gradient
            if not get_accelerator().is_synchronized_device():
                param.grad.record_stream(get_accelerator().current_stream())
            param.grad = None

        if self.offload_optimizer and self.swap_optimizer:
            for i in offload_fp32_gradients.keys():
                self.optimizer_swapper.swap_out_gradients(parameter=self.fp32_partitioned_groups_flat[i],
                                                          gradient_offsets=offload_fp32_offsets[i],
                                                          gradient_tensors=offload_fp32_gradients[i])
        see_memory_usage(f"after partition_grads", force=ENBALE_MEM_DEBUG)
        return buffers

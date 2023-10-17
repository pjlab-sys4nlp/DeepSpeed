import os
import gc
import psutil

# if global_zero35_manager is None:
global_zero35_manager = None

from deepspeed.accelerator import get_accelerator
from deepspeed import comm as dist
from deepspeed.utils import logger
from deepspeed.runtime.swap_tensor.partitioned_param_swapper import PartitionedParamStatus
from deepspeed.utils import instrument_w_nvtx, logger
from typing import Callable, Iterable
from torch.nn import Parameter


import torch

init_count = 0

FORCE = False

torch_memory_reserved = get_accelerator().memory_reserved
torch_max_memory_reserved = get_accelerator().max_memory_reserved

# avoid circular reference
def see_memory_usage(message, force=False):
    if not force:
        return
    if dist.is_initialized() and not dist.get_rank() == 0:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # Print message except when distributed but not rank 0
    logger.info(message)
    logger.info(f"MA {round(get_accelerator().memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
        Max_MA {round(get_accelerator().max_memory_allocated() / (1024 * 1024 * 1024),2)} GB \
        CA {round(torch_memory_reserved() / (1024 * 1024 * 1024),2)} GB \
        Max_CA {round(torch_max_memory_reserved() / (1024 * 1024 * 1024))} GB ")

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    logger.info(f'CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%')

    # get the peak memory to report correct data, so reset the counter for the next call
    get_accelerator().reset_peak_memory_stats()


def zero35_debug(msg, rank=None, force=FORCE, flush=True):
    if force:
        msg = f"Rank: {os.environ['SLURM_PROCID']}, " + msg
        if rank is None:
            if flush:
                print(msg, flush=True)
            else:
                logger.info(msg)
        elif os.environ['SLURM_PROCID'] == str(rank):
            if flush:
                print(msg, flush=True)
            else:
                logger.info(msg)


def zero35_g_p_reduce_scatter_coalesced(tensor_list, partition_type):
    # reshape 的逻辑
    # [0, 1, 2, 3, 4, 5, 6, 7]  -> [0, 4, 1, 5, 2, 6, 3, 7]
    # [0, 1, 2, 3, 4, 5, 6, 7]  -> [0, 4, 1, 5, 2, 6, 3, 7]
    # [0, 1, 2, 3, 4, 5, 6, 7]  -> [0, 4, 1, 5, 2, 6, 3, 7]
    # [0, 1, 2, 3, 4, 5, 6, 7]  -> [0, 4, 1, 5, 2, 6, 3, 7]

    see_memory_usage(f"before zero35_g_p_reduce_scatter_coalesced, partition_type:{partition_type}")

    do_reshape = partition_type == "grad" or partition_type == "param"
    dtype = tensor_list[0].dtype

    # if do_reshape:
    dp_comm_group = global_zero35_manager._dp_process_group
    param_comm_group = global_zero35_manager._param_process_group

    if do_reshape:
        scatter_comm_group = param_comm_group

        # dp_world_size = dist.get_world_size(dp_comm_group)
        # param_world_size = dist.get_world_size(param_comm_group)

        # new_tensor_list = []
        # _undo_indexs_for_per_tensor = []
        # for grad in tensor_list:
        #     assert grad.numel() % dp_world_size == 0
        #     assert grad.numel() % param_world_size == 0

        #     dp_partition_size = int(grad.numel() / dp_world_size)  # 按照dp范围划分的最小part大小
        #     param_partition_size = int(grad.numel() / param_world_size)
        #     assert param_partition_size % dp_partition_size == 0
        #     param_partition_num = int(param_partition_size / dp_partition_size)  # 每个节点内包含的 dp_partition_size 的数量
        #     grad = grad.reshape(-1, dp_partition_size)
        #     indexs = []
        #     for idx in range(param_world_size):
        #         for jdx in range(param_partition_num):
        #             indexs.append(idx + jdx * param_world_size)

        #     # zero35_debug(f"scatter index : {indexs}")

        #     indexs=torch.tensor(indexs).to(get_accelerator().device_name())
        #     _, undo_indices = torch.sort(indexs, dim=0, descending=False)
        #     _undo_indexs_for_per_tensor.append(undo_indices)

        #     reshape_grad = torch.index_select(grad, 0, indexs)
        #     assert reshape_grad.is_contiguous()
        #     new_tensor_list.append(reshape_grad.view(-1))
        # tensor_list = new_tensor_list
    else:
        scatter_comm_group = dp_comm_group

    # if do_reshape:
    #     new_tensor_list = []
    #     for i, grad in enumerate(tensor_list):
    #         new_tensor_list.append(torch.index_select(grad, 0, _undo_indexs_for_per_tensor[i]))
    #     tensor_list = new_tensor_list

    see_memory_usage(f"after zero35_g_p_reduce_scatter_coalesced, partition_type:{partition_type}")

    return tensor_list, scatter_comm_group

def zero35_g_p_all_gather_coalesced(tensor_list, partition_type=None):
    # reshape 的逻辑
    # [0, 4, 1, 5, 2, 6, 3, 7]  -> [0, 1, 2, 3, 4, 5, 6, 7]  
    # [0, 4, 1, 5, 2, 6, 3, 7]  -> [0, 1, 2, 3, 4, 5, 6, 7] 
    # [0, 4, 1, 5, 2, 6, 3, 7]  -> [0, 1, 2, 3, 4, 5, 6, 7] 
    # [0, 4, 1, 5, 2, 6, 3, 7]  -> [0, 1, 2, 3, 4, 5, 6, 7]
    # 
    # [0, , 1, ] 
    # do_reshape = partition_type == "grad" or partition_type == "param"
    dtype = tensor_list[0].dtype

    # if do_reshape:
    see_memory_usage(f"before zero35_g_p_all_gather_coalesced, partition_type:{partition_type}")

    dp_comm_group = global_zero35_manager._dp_process_group
    param_comm_group = global_zero35_manager._param_process_group

    all_gather_comm_group = param_comm_group

    # dp_world_size = dist.get_world_size(dp_comm_group)
    # param_world_size = dist.get_world_size(param_comm_group)

    # new_tensor_list = []

    # for t_data in tensor_list:
    #     param_full_tensor = t_data.data
    #     indexs = []

    #     partition_unit_size = t_data.numel() // dp_world_size  # 按照dp范围划分的最小part大小

    #     # zero35_debug(f"param_full_tensor.numel() :{t_data.numel()}, dp_world_size:{dp_world_size},partition_unit_size:{partition_unit_size}", flush=True)

    #     partition_unit_num = t_data.numel() // partition_unit_size

    #     partition_unit_size_per_rank = t_data.numel() // param_world_size  
    #     partition_unit_num_per_rank = partition_unit_size_per_rank // partition_unit_size # 2
    #     partition_unit_num_per_node = partition_unit_num_per_rank * param_world_size    # 2 * 4 -> 8

    #     param_full_tensor = param_full_tensor.reshape(-1, partition_unit_size)

    #     for idx in range(partition_unit_num_per_rank): # 8
    #         indexs.extend([idx + jdx * partition_unit_num_per_rank for jdx in range(param_world_size)])

    #     # zero35_debug(f"gather index : {indexs}")
    #     # indexs = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]
    #     indexs=torch.tensor(indexs).to(get_accelerator().device_name())
    #     reshape_t_data = torch.index_select(param_full_tensor, 0, indexs)
    #     reshape_t_data = reshape_t_data.view(t_data.data.shape)
    #     assert reshape_t_data.is_contiguous()

    #     # param_full_tensor.ds_tensor = reshape_t_data
    #     t_data.data = reshape_t_data

    #     new_tensor_list.append(reshape_t_data)
    # tensor_list = new_tensor_list
    # # else:
    # #     all_gather_comm_group = dp_comm_group
    see_memory_usage(f"before zero35_g_p_all_gather_coalesced, partition_type:{partition_type}")

    return tensor_list, all_gather_comm_group


class GlobalZero35GroupManager:
    def __init__(self, enable_zero35, mpu=None) -> None:
        print(f"enable_zero35: {enable_zero35} !!!", flush=True)
        # TODO, zero35 不能和tp或pp一起使用
        self.enable_zero35 = enable_zero35
        self._dp_process_group = dist.get_world_group()
        self._os_rank = dist.get_rank()
        self.hierarchical_allgather = 'hierarchical' in os.environ
        self.world_size = dist.get_world_size(self._dp_process_group)
        self.local_device = torch.device(get_accelerator().device_name(os.environ["LOCAL_RANK"]))

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

        self.zero35_parallel_size = 8 # TODO: 将这个变成可配置参数
        self.rank_num_per_dp_group = self.world_size // self.data_parallel_size
        self.num_zero35_parallel_group = self.data_parallel_size // self.zero35_parallel_size

        if self.enable_zero35:
            self.zero35_group = self.init_zero35_process_group()
            self.zero35_hierarchical_group = self.init_zero35_hierarchical_process_group()
            self._grad_process_group = self.zero35_group
            self._param_process_group = self.zero35_group

            self._param_rank = dist.get_rank(group=self._param_process_group)
            self._param_world_size = dist.get_world_size(group=self._param_process_group)

            self._grad_rank = dist.get_rank(group=self._grad_process_group)
            self._grad_world_size = dist.get_world_size(group=self._grad_process_group)

        else:
            self.zero35_group = None
            self.zero35_hierarchical_group = None
            self._grad_process_group = self._dp_process_group
            self._param_process_group = self._dp_process_group
            
            self._param_rank = 0
            self._param_world_size = 1
            self._grad_rank = 0
            self._grad_world_size = 1
        
        if dist.get_rank() < 8:
            print(f"Rank:{dist.get_rank()} zero35_hierarchical_group: {dist.get_all_ranks_from_group(self.zero35_hierarchical_group)}", flush=True)
        
        if self.hierarchical_allgather:
            assert self._param_world_size == 8, "hierarchical_allgather only support param shared 8!"

        zero35_debug(f"zero35_parallel_size: {self.zero35_parallel_size}, num_zero35_parallel_group:{self.num_zero35_parallel_group}, ranks:{dist.get_all_ranks_from_group(self.zero35_group)}", force=True)


    def get_partition_dp_group(self, param, partition_type):
        """ Return the communication group with all data-parallel ranks """
        if partition_type == "os":
            return param.dp_process_group
        elif partition_type == "grad":
            return param.grad_process_group
        elif partition_type == "param":
            return param.param_process_group
        else:
            assert False, f"unkonwn partition_type: {partition_type}"

    def get_partition_rank(self, partition_type):
        """subclass can overload to specify different relative rank in
        parameter partition group"""
        if partition_type == "os":
            return self._os_rank
        elif partition_type == "grad":
            return self._grad_rank
        elif partition_type == "param":
            return self._param_rank
        else:
            assert False, f"unkonwn partition_type: {partition_type}"

    def get_dp_process_group(self, partition_type):
        """ Return the communication group with all data-parallel ranks """
        if partition_type == "os":
            return self._dp_process_group
        elif partition_type == "grad":
            return self._grad_process_group
        elif partition_type == "param":
            return self._param_process_group
        else:
            assert False, f"unkonwn partition_type: {partition_type}"

    def get_rank_in_group(self, partition_type):
        if partition_type == "os":
            return self._os_rank
        elif partition_type == "grad":
            return self._grad_rank
        elif partition_type == "param":
            return self._param_rank
        else:
            assert False, f"unkonwn partition_type: {partition_type}"

    def get_world_size(self, partition_type):
        return dist.get_world_size(self.get_dp_process_group(partition_type))

    def get_partition_unit_size(self, param_size):
        dp_partition_count = self.num_partitions("os")    
        assert param_size % dp_partition_count == 0, f"param realy size: {param_size}, dp_partition_count: {dp_partition_count}"
        return param_size // dp_partition_count

    def init_zero35_process_group(self):
        my_zero35_group = None

        for i in range(self.model_parallel_size):
            for j in range(self.num_zero35_parallel_group):
                ranks = [
                    i + (j * self.zero35_parallel_size + k) * self.model_parallel_size
                    for k in range(self.zero35_parallel_size)
                ]
                group = dist.new_group(ranks)

                if dist.get_rank() in ranks:
                    my_zero35_group = group

        return my_zero35_group

    def init_zero35_hierarchical_process_group(self):
        my_hierarchical_zero35_group = None
        gpus_per_node = 8
        nnodes = self.world_size // 8

        for i in range(gpus_per_node):
            ranks = [i + 8*j for j in range(nnodes)]
            group = dist.new_group(ranks)
            if dist.get_rank() in ranks:
                my_hierarchical_zero35_group = group
        
        return my_hierarchical_zero35_group


    def get_sub_p_g_parition(self, ds_param, grad=None):
        assert hasattr(ds_param, 'ds_numel'), 'get_sub_p_g_parition input must be ds_param'
        if self.enable_zero35:
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
        else:
            if grad is not None:
                return grad
            else:
                return ds_param.ds_tensor

    def zero35_judge_gahter_boundary(self, mico_step, forward):
        return mico_step == 0 and forward
            
    def zero35_hack_allgahter_ds_tensor(self, param, mico_step, forward):

        see_memory_usage(f"before zero35_hack_allgahter_ds_tensor, mico_step:{mico_step}, forward:{forward}")
        if self.zero35_judge_gahter_boundary(mico_step, forward):
            # gather boundary
            partition_type = "os"

            assert hasattr(param, 'ds_numel'), 'zero35_hack_allgahter_ds_tensor input must be ds_param'
            parition_num = self.get_world_size(partition_type)

            node_id =  dist.get_rank() // self.zero35_parallel_size
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


    def zero35_restore_allgahter_ds_tensor(self, __param):
        # TODO:(wgt)
        if __param.ds_tensor.is_first_fwd_all_gahter == True:
            # zero35_debug(f"now ds_tensor numel: {__param.ds_tensor.ds_numel}, backup numel: {__param.ds_numel_backup}")
            # zero35_debug(f"now ds_tensor data: {__param.ds_tensor.data}, backup data: {__param.ds_tensor_backup}")

            __param.ds_tensor.data = __param.ds_tensor_backup
            __param.ds_tensor.is_first_fwd_all_gahter = False
            __param.ds_tensor.ds_numel = __param.ds_numel_backup
        return __param

    # @property
    def num_partitions(self, partition_type):
        return dist.get_world_size(group=self.get_dp_process_group(partition_type))


    @instrument_w_nvtx
    def zero35_hierarchical_all_gather_params(self, params: Iterable[Parameter],
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
                param_chunk = param_tensors[i].view((inter_node_size, intra_node_size, p.ds_tensor.ds_numel)).narrow(1, local_rank, 1)
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


def get_global_zero35_manager(enable_zero35=None, mpu=None) -> GlobalZero35GroupManager:
    global global_zero35_manager
    global init_count
    if global_zero35_manager is None:
        print(f"init GlobalZero35GroupManager!!, init_count: {init_count}", flush=True)
        init_count += 1
        if enable_zero35 is None:
            enable_zero35 = 'enable_zero35' in os.environ
        global_zero35_manager = GlobalZero35GroupManager(enable_zero35=enable_zero35, mpu=mpu)
    else:
        return global_zero35_manager

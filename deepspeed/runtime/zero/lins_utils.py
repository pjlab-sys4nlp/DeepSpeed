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
from contextlib import contextmanager

import torch

init_count = 0

FORCE = False

torch_memory_reserved = get_accelerator().memory_reserved
torch_max_memory_reserved = get_accelerator().max_memory_reserved

# global param process group
global_lins_utils = None


class LinSProcessGroup:

    def __init__(self, dp_process_group, lins_param_partition_num: int, lins_grad_partition_num: int,
                 lins_os_partition_num: int) -> None:

        self.world_size = dist.get_world_size(dp_process_group)
        self.data_parallel_size = self.world_size
        self.model_parallel_size = 1  # not support model parallel

        self.lins_param_partition_num = lins_param_partition_num
        self.lins_grad_partition_num = lins_grad_partition_num
        if lins_os_partition_num == -1:
            self.lins_os_partition_num = self.data_parallel_size

        assert lins_param_partition_num == lins_grad_partition_num and lins_grad_partition_num == 8, \
            "Only test lins_param_partition_num==lins_grad_partition_num == 8"
        assert self.lins_os_partition_num == self.data_parallel_size, \
            f"Only test lins_os_partition_num:{self.lins_os_partition_num} == self.data_parallel_size:{self.data_parallel_size}"

        self.zero35_hierarchical_group = self.init_zero35_hierarchical_process_group()
        self._grad_process_group = self.init_lins_process_group(
            self.lins_grad_partition_num, self.data_parallel_size // self.lins_grad_partition_num)
        self._param_process_group = self.init_lins_process_group(
            self.lins_param_partition_num, self.data_parallel_size // self.lins_param_partition_num)

        self._param_rank = dist.get_rank(group=self._param_process_group)
        self._param_world_size = dist.get_world_size(group=self._param_process_group)

        self._grad_rank = dist.get_rank(group=self._grad_process_group)
        self._grad_world_size = dist.get_world_size(group=self._grad_process_group)

        # if dist.get_rank() < 8:
        print(f"Rank:{dist.get_rank()} \
zero35_hierarchical_group: {dist.get_all_ranks_from_group(self.zero35_hierarchical_group)}, \
zero35_group:{dist.get_all_ranks_from_group(self._grad_process_group)} \
self._param_rank: {self._param_rank}, \
self._param_world_size: {self._param_world_size}, \
self._grad_rank {self._grad_rank}, \
self._grad_world_size: {self._grad_world_size}",
              flush=True)

    def init_lins_process_group(self, parallel_size, parallel_num):
        my_zero35_group = None

        for i in range(self.model_parallel_size):
            for j in range(parallel_num):
                ranks = [i + (j * parallel_size + k) * self.model_parallel_size for k in range(parallel_size)]
                group = dist.new_group(ranks)

                if dist.get_rank() in ranks:
                    my_zero35_group = group

        return my_zero35_group

    def init_zero35_hierarchical_process_group(self):
        my_hierarchical_zero35_group = None
        gpus_per_node = 8
        nnodes = self.world_size // 8

        for i in range(gpus_per_node):
            ranks = [i + 8 * j for j in range(nnodes)]
            group = dist.new_group(ranks)
            if dist.get_rank() in ranks:
                my_hierarchical_zero35_group = group

        return my_hierarchical_zero35_group


global_parition_type = {"type": "os"}


@contextmanager
def set_lins_parition_type(partition_type):
    global global_parition_type
    old_parition_type = global_parition_type["type"]
    try:
        global_parition_type["type"] = partition_type
        yield
    finally:
        global_parition_type["type"] = old_parition_type


def zero35_judge_gahter_boundary(mico_step, forward):
    """判断是不是需要做dp范围的 all-gahter

    Args:
        mico_step (_type_): _description_
        forward (_type_): _description_

    Returns:
        _type_: _description_
    """
    return mico_step == 0 and forward


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

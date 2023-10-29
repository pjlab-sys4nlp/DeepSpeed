import torch
import os
import torch.nn.functional as F

from torch.distributed import GroupMember
import torch.nn as nn
import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from torch.optim import Adam
from deepspeed.comm import init_distributed
import torch.distributed as dist
import numpy as np
from torch.utils.data import Dataset

json_path = "stage3.json"


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def get_master_node():
    import subprocess

    if os.getenv("SLURM_JOB_ID") is None:
        raise RuntimeError("get_master_node can only used in Slurm launch!")
    result = subprocess.check_output('scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1', shell=True)
    result = result.decode("utf8").strip()
    return result


class ToyModel2(torch.nn.Module):

    def __init__(self) -> None:
        super(ToyModel2, self).__init__()
        self.step_count = 0
        self.dense1 = torch.nn.Parameter(
            data=torch.tensor([-i for i in range(64)], dtype=torch.bfloat16, requires_grad=True))

    def forward(self, x, labels=None):
        x = x.to(f"cuda:{int(os.environ['SLURM_PROCID']) % 8}")
        if os.environ['SLURM_PROCID'] == '0':
            print(f"Rank: {os.environ['SLURM_PROCID']}:\
do ToyModel2-forward!,self.step_count: {self.step_count}, \
self.dense:{self.dense1}",
                  flush=True)

        self.step_count += 1
        y = self.dense1.mul(x)
        # x =  self.dense2.mul(y)
        return y

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor


class ToyModel(torch.nn.Module):

    def __init__(self) -> None:
        # 64 / 16 = 4
        # 64 / 8 = 8
        super(ToyModel, self).__init__()
        self.step_count = 0
        # self.dense = torch.nn.Linear(64, 1, bias=False, dtype=torch.bfloat16)

        # 我们让 reduce_bucket_size 正好等于 64，这样在做完 ToyModel2 的 bwd 后就可以直接进行 bucket 的 allreduce
        self.dense1 = torch.nn.Parameter(
            data=torch.tensor([i for i in range(64)], dtype=torch.bfloat16, requires_grad=True))
        self.sub_module = ToyModel2()
        self.dense2 = torch.nn.Parameter(
            data=torch.tensor([-i for i in range(64)], dtype=torch.bfloat16, requires_grad=True))

    def forward(self, x, labels=None):
        x = x.to(f"cuda:{int(os.environ['SLURM_PROCID']) % 8}")
        if os.environ['SLURM_PROCID'] == '0':
            print(f"Rank: {os.environ['SLURM_PROCID']}:\
do forward!,self.step_count: {self.step_count}, \
self.dense:{self.dense1}",
                  flush=True)

        self.step_count += 1
        y = self.sub_module(x)
        y = self.dense1 + y
        x = self.dense2.mul(y)
        return x

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building llama model ...')
    see_memory_usage(f"Before Building Model", force=True)


    init_distributed(dist_backend="nccl", \
                    auto_mpi_discovery=False, \
                    init_method=f"tcp://[{get_master_node()}]:12349", \
                    rank=int(os.environ['SLURM_PROCID']), \
                    world_size=16)

    local_group1 = dist.new_group([i for i in range(8)])
    local_group2 = dist.new_group([i for i in range(8, 16, 1)])

    os.environ['LOCAL_RANK'] = f"{int(os.environ['SLURM_PROCID']) % 8}"

    def get_param_group():
        if int(os.environ['SLURM_PROCID']) < 8:
            return local_group1
        else:
            return local_group2

    from deepspeed.runtime.zero.lins import LinS_Init
    with LinS_Init(data_parallel_group=GroupMember.WORLD,
                   remote_device=None,
                   config_dict_or_path=json_path,
                   enabled=True,
                   mpu=None):
        model = ToyModel()
    see_memory_usage(f"After Building Model", force=True)
    return model


def _get_params_for_weight_decay_optimization_one_subgourp(modules):
    weight_decay_params = {'params': [], 'name': 'weight_decay_params'}
    for module in modules:
        for module_ in module.modules():
            for num, param in list(module_._parameters.items()):
                weight_decay_params['params'].extend([param])

    return weight_decay_params,


class RandomDataset(torch.utils.data.IterableDataset):

    def __init__(self, num_samples=1000000, seq_len=1024) -> None:
        super().__init__()
        self.len = 1024
        self.data = [i for i in range(64)]

    def __getitem__(self, index):
        return np.array(self.data, dtype=int)

    def get_dataset_name(self):
        return "test_toy"

    def __len__(self):
        return self.len


class SimpleBatchSampler:

    def __init__(self, total_samples) -> None:
        self.total_samples = total_samples

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        yield list(range(0, 64))


# srun -p llm_t --quotatype=spot  -n16 -N2 --ntasks-per-node=8 --gpus-per-task=1 python  test_lins.py
if __name__ == "__main__":
    model = model_provider()
    model = [model]
    param_groups = _get_params_for_weight_decay_optimization_one_subgourp(model)
    optimizer = Adam(param_groups, lr=1e-4, weight_decay=1e-2, betas=(1e-2, 1e-2), eps=1e-2)

    lr_scheduler = None
    train_ds = RandomDataset()
    # batch_sampler = SimpleBatchSampler(total_samples=1024)
    # train_dataloader = torch.utils.data.DataLoader(train_ds,
    #                                                batch_sampler=None,
    #                                                num_workers=0,
    #                                                pin_memory=False)
    (
        model,
        optimizer,
        deepspeed_dataloader,
        lr_scheduler,
    ) = deepspeed.initialize(
        model=model[0],
        optimizer=optimizer,
        args=None,
        lr_scheduler=lr_scheduler,
        training_data=train_ds,
        mpu=None,
        config=json_path,
    )

    data_iterator = iter(deepspeed_dataloader)
    for i in range(10):
        loss = model[0].train_batch(data_iter=data_iterator)

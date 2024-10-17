import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import numpy as np
import torch
from torch.utils.data import (
    BatchSampler,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data.distributed import DistributedSampler

# Adapted from MixedBatchSampler 

class DistributedMixedBatchSampler(BatchSampler):
    """Sample one batch from a selected dataset with given probability.
    Compatible with datasets at different resolution
    """

    def __init__(
        self, src_dataset_ls, batch_size, drop_last, shuffle, world_size, rank, prob=None, generator=None
    ):
        self.base_sampler = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

        self.src_dataset_ls = src_dataset_ls
        self.n_dataset = len(self.src_dataset_ls)

        # Dataset length
        self.dataset_length = [len(ds) for ds in self.src_dataset_ls]
        self.cum_dataset_length = np.cumsum([0] + self.dataset_length[:-1])  # cumulative dataset length
        
        self.src_batch_samplers = [
            BatchSampler(
                sampler=DistributedSampler(
                    ds, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
                ),
                batch_size=self.batch_size,
                drop_last=self.drop_last,
            )
            for ds in self.src_dataset_ls
        ]
        self.raw_batches = [
            list(bs) for bs in self.src_batch_samplers
        ]  # index in original dataset
        self.n_batches = [len(b) for b in self.raw_batches]
        self.n_total_batch = sum(self.n_batches)

        # sampling probability
        if prob is None:
            # if not given, decide by dataset length
            self.prob = torch.tensor(self.n_batches) / self.n_total_batch
        else:
            self.prob = torch.as_tensor(prob)

    def __iter__(self):
        """_summary_

        Yields:
            list(int): a batch of indics, corresponding to ConcatDataset of src_dataset_ls
        """
        for _ in range(self.n_total_batch):
            idx_ds = torch.multinomial(
                self.prob, 1, replacement=True, generator=self.generator
            ).item()
            # if batch list is empty, generate new list
            if 0 == len(self.raw_batches[idx_ds]):
                self.raw_batches[idx_ds] = list(self.src_batch_samplers[idx_ds])
            # get a batch from list
            batch_raw = self.raw_batches[idx_ds].pop()
            # shift by cumulative dataset length
            shift = self.cum_dataset_length[idx_ds]
            batch = [n + shift for n in batch_raw]
            yield batch
    
    def set_epoch(self, epoch):
        for sampler in self.src_batch_samplers:
            if isinstance(sampler.sampler, DistributedSampler):
                sampler.sampler.set_epoch(epoch)

    def __len__(self):
        return self.n_total_batch

# Unit test
if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader, ConcatDataset

    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group("nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        print("Distributed training not initialized. Running in single GPU mode.")
        world_size = 1
        rank = 0

    if dist.is_initialized():
        assert 20 % world_size == 0, "Batch size must be divisible by world size."
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
    else:
        device = torch.cuda.current_device()

    class SimpleDataset(Dataset):
        def __init__(self, start, length) -> None:
            super().__init__()
            self.start = start
            self.len = length

        def __len__(self):
            return self.len

        def __getitem__(self, index):
            return self.start + index

    dataset_1 = SimpleDataset(0, 20)
    dataset_2 = SimpleDataset(1000, 20)
    dataset_3 = SimpleDataset(10000, 20)

    concat_dataset = ConcatDataset([dataset_1, dataset_2, dataset_3])

    mixed_sampler = DistributedMixedBatchSampler(
        src_dataset_ls=[dataset_1, dataset_2, dataset_3],
        batch_size=4,
        drop_last=True,
        shuffle=True,
        world_size=world_size,
        rank=rank,
        prob=[0.6, 0.3, 0.1],
        generator=torch.Generator().manual_seed(0),
    )

    mixed_sampler.set_epoch(1)
    loader = DataLoader(concat_dataset, batch_sampler=mixed_sampler)

    print(f"Rank {rank}: Total number of batches: {len(loader)}")
    for i, d in enumerate(loader):
        print(f"Rank {rank}, Batch {i+1}: {d.tolist()}")

    if dist.is_initialized():
        dist.destroy_process_group()


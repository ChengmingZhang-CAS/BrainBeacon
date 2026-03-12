from torch.utils.data import Sampler
import torch
from collections import defaultdict
import random


class HierarchicalDistributedSampler(Sampler):
    def __init__(self, dataset, label_fn, label_weights, num_samples, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset)
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()

        self.dataset = dataset
        self.label_fn = label_fn
        self.label_weights = label_weights
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0
        self.num_samples = num_samples

        # 构建 label -> sample_idx 列表
        self.label_to_indices = defaultdict(list)
        for i in range(len(dataset)):
            label = label_fn(i)
            self.label_to_indices[label].append(i)

        # label list & 权重张量
        self.labels = list(self.label_to_indices.keys())
        raw_weights = torch.tensor([label_weights.get(lbl, 1.0) for lbl in self.labels], dtype=torch.float)
        self.label_probs = raw_weights / raw_weights.sum()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.rank)

        # 每张卡采样数量
        total_size = self.num_samples * self.num_replicas
        label_samples = torch.multinomial(self.label_probs, total_size, replacement=True)

        indices = []
        for lbl_idx in label_samples.tolist():
            label = self.labels[lbl_idx]
            pool = self.label_to_indices[label]
            idx = random.choice(pool)
            indices.append(idx)

        # 分发当前rank的子集
        indices = indices[self.rank:total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

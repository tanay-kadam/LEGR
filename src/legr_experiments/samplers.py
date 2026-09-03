from __future__ import annotations

from collections import defaultdict
import math
import random

from torch.utils.data import Sampler


class GroupAwareBatchSampler(Sampler[list[int]]):
    """Batches distinct DAGs while deliberately co-locating existing twins."""

    def __init__(self, samples, batch_size: int, seed: int = 42, drop_last: bool = True):
        self.samples = samples
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.by_dag = defaultdict(list)
        self.dags_by_group = defaultdict(list)
        for idx, sample in enumerate(samples):
            self.by_dag[sample.dag_index].append(idx)
        for dag_index, indices in self.by_dag.items():
            group = samples[indices[0]].group_index
            self.dags_by_group[group].append(dag_index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        count = len(self.by_dag)
        return count // self.batch_size if self.drop_last else math.ceil(count / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        groups = list(self.dags_by_group)
        rng.shuffle(groups)
        selected = []
        for group in groups:
            dags = list(self.dags_by_group[group])
            rng.shuffle(dags)
            for dag in dags:
                selected.append(rng.choice(self.by_dag[dag]))
        for start in range(0, len(selected), self.batch_size):
            batch = selected[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

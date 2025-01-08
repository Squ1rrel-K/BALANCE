import numpy as np
import torch
import torch.utils.data as data
from typing import Any

from util import to_array


class NonIIDDataset(data.Dataset):

    def __init__(self, dataset: data.Dataset):
        self.id = None
        self.dataset = dataset

    def shuffle(self, alpha, n_clients):
        self.id = dirichlet_split_noniid(to_array(self.dataset.targets, 0), alpha, n_clients)
        self.id = np.concatenate(self.id)

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            (Any): Sample and metadata, optionally transformed by the respective transforms.
        """
        if self.id is not None:
            return self.dataset.__getitem__(self.id[index])
        return self.dataset.__getitem__(index)

    def __len__(self) -> int:
        return self.dataset.__len__()

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    n_classes = train_labels.max() + 1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs
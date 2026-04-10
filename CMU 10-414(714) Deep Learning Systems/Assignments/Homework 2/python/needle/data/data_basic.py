import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.index = 0
        if self.shuffle:
          self.ordering = np.arange(len(self.dataset))
          # 打乱整个数据集
          np.random.shuffle(self.ordering)
          self.ordering = np.array_split(self.ordering, range(self.batch_size, len(self.dataset), self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.index >= len(self.ordering):
          raise StopIteration
        # Pythonic 的列表推导式, 速度稍快
        data = [self.dataset[i] for i in self.ordering[self.index]]
        self.index+=1
        result = []
        # 将样本列表 [(img1, lbl1), (img2, lbl2), ...] 转换成了类别组 [(img1, img2, ...), (lbl1, lbl2, ...)]
        for data_type_group in zip(*data):
            # 将 numpy 数组列表 stack 起来变成一个大数组
            # 沿着指定的轴将多个数组堆叠起来，从而创建一个新的多维数组
            batched_ndarray = np.stack(data_type_group)
            # 转换为 Needle Tensor 并存入结果
            result.append(Tensor(batched_ndarray))
            
        return tuple(result)
        ### END YOUR SOLUTION


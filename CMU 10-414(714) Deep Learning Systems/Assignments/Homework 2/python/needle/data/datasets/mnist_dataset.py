from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(label_filename, 'rb') as f:
          magicNumber, itemsNumber = struct.unpack('>ii', f.read(8))
          self.labels = np.frombuffer(f.read(), dtype=np.uint8)

        with gzip.open(image_filename, 'rb') as f:
          magicNumber, itemsNumber, rowNumber, colNumber = struct.unpack('>iiii', f.read(16))
          images = np.frombuffer(f.read(), dtype=np.uint8)
          images = images.reshape(itemsNumber, rowNumber, colNumber, 1)
          self.images = images.astype(np.float32) / 255

        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # 确保 Data Augmentation（数据增强）是在副本上进行的，不会污染原始存储在内存中的 self.images 数据集
        img = np.array(self.images[index])
        if self.transforms is not None:
          for transform in self.transforms:
            img = transform(img)
        return img, self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION
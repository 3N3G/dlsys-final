import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.base_folder = base_folder
        self.train = train
        self.p = p
        self.transforms = transforms or []

        if train:
            files = [f"data_batch_{i}" for i in range(1, 5 + 1)]
        else:
            files = ["test_batch"]

        imgs = []
        labels = []

        def _load_pickle(fp):
            with open(fp, "rb") as f:
                return pickle.load(f, encoding="latin1")

        for fname in files:
            path = os.path.join(base_folder, fname)
            d = _load_pickle(path)
            data = d.get("data", None)
            if data is None:
                data = d.get(b"data")
            lbs = d.get("labels", None)
            if lbs is None:
                lbs = d.get(b"labels")

            data = np.asarray(data, dtype=np.float32)
            n = data.shape[0]
            data = data.reshape(n, 3, 32, 32)
            data = data / 255.0

            imgs.append(data)
            labels.append(np.asarray(lbs, dtype=np.int64))

        self.X = np.concatenate(imgs, axis=0) if len(imgs) > 1 else imgs[0]
        self.y = np.concatenate(labels, axis=0) if len(labels) > 1 else labels[0]
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        img_chw = self.X[index]
        label = int(self.y[index])

        if self.transforms:
            img_hwc = np.transpose(img_chw, (1, 2, 0))
            for t in self.transforms:
                img_hwc = t(img_hwc)
            if img_hwc.ndim == 3 and img_hwc.shape == (32, 32, 3):
                img_chw = np.transpose(img_hwc, (0, 3, 1, 2)) if img_hwc.ndim == 4 else np.transpose(img_hwc, (2, 0, 1))
            elif img_hwc.ndim == 3 and img_hwc.shape == (3, 32, 32):
                img_chw = img_hwc
            else:
                raise ValueError(f"Transform returned image with shape {img_hwc.shape}; expected (32,32,3) or (3,32,32).")

        img_chw = img_chw.astype(np.float32, copy=False)
        return img_chw, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION

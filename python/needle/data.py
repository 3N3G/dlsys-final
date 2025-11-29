"""Data loading utilities for CIFAR-10 and PTB datasets."""
import os
import pickle
import numpy as np
from typing import Iterator, Optional, List, Sized, Union, Iterable
import gzip
import struct
from . import Tensor
from . import backend_ndarray as nd


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C NDArray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            return np.flip(img, axis=1)
        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C NDArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        H, W, C = img.shape
        padded = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        start_x = self.padding + shift_x
        start_y = self.padding + shift_y
        return padded[start_x:start_x + H, start_y:start_y + W, :]


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
        device=None,
        dtype="float32"
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )
        self.device = device
        self.dtype = dtype

    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            indices = np.arange(len(self.dataset))
            np.random.shuffle(indices)
            self.ordering = np.array_split(
                indices, range(self.batch_size, len(self.dataset), self.batch_size)
            )
        return self

    def __next__(self):
        if self.idx >= len(self.ordering):
            raise StopIteration

        batch_indices = self.ordering[self.idx]
        self.idx += 1

        # Collect batch
        batch = [self.dataset[i] for i in batch_indices]

        # Separate X and y
        X_list = [x for x, y in batch]
        y_list = [y for x, y in batch]

        # Stack into arrays
        X_batch = np.stack(X_list, axis=0)
        y_batch = np.array(y_list)

        # Convert to Tensors
        X_tensor = Tensor(X_batch, device=self.device, dtype=self.dtype)
        # Note: backend only supports float32, so we use float32 for labels too
        y_tensor = Tensor(y_batch, device=self.device, dtype=self.dtype)

        return X_tensor, y_tensor


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
        super().__init__(transforms)
        self.base_folder = base_folder
        self.train = train
        self.p = p

        # Load data
        if train:
            # Load training batches
            data_batches = []
            label_batches = []
            for i in range(1, 6):
                filepath = os.path.join(base_folder, f"data_batch_{i}")
                with open(filepath, 'rb') as f:
                    batch_dict = pickle.load(f, encoding='bytes')
                    data_batches.append(batch_dict[b'data'])
                    label_batches.append(batch_dict[b'labels'])

            self.X = np.concatenate(data_batches, axis=0)
            self.y = np.concatenate(label_batches, axis=0)
        else:
            # Load test batch
            filepath = os.path.join(base_folder, "test_batch")
            with open(filepath, 'rb') as f:
                batch_dict = pickle.load(f, encoding='bytes')
                self.X = batch_dict[b'data']
                self.y = np.array(batch_dict[b'labels'])

        # Reshape and normalize
        # X is (N, 3072) -> (N, 3, 32, 32) in CHW format
        self.X = self.X.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        self.y = self.y.astype(np.uint8)

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        img = self.X[index]  # Shape: (3, 32, 32)
        label = self.y[index]

        # Apply transforms if in training mode (transforms expect HWC format)
        if self.transforms:
            # Convert CHW -> HWC for transforms
            img_hwc = np.transpose(img, (1, 2, 0))  # (32, 32, 3)
            img_hwc = self.apply_transforms(img_hwc)
            # Convert back HWC -> CHW
            img = np.transpose(img_hwc, (2, 0, 1))  # (3, 32, 32)

        return img, label

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return len(self.X)


class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        return len(self.idx2word)


class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """

    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, "train.txt"), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, "test.txt"), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ids = []
        with open(path, 'r') as f:
            for line_num, line in enumerate(f):
                if max_lines is not None and line_num >= max_lines:
                    break
                words = line.split() + ['<eos>']
                for word in words:
                    ids.append(self.dictionary.add_word(word))
        return ids


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    # Trim off remainder
    num_batches = len(data) // batch_size
    data = data[:num_batches * batch_size]
    # Reshape into (batch_size, nbatch) then transpose to (nbatch, batch_size)
    data_array = np.array(data).reshape(batch_size, num_batches).T
    return data_array


def get_batch(data, i, bptt, device=None, dtype="float32"):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    data - numpy array returned from batchify function of shape (nbatch, batch_size)
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, batch_size) with cached data as NDArray
    target - Tensor of shape (bptt*batch_size,) with cached data as NDArray
    """
    nbatch, batch_size = data.shape
    seq_len = min(bptt, nbatch - 1 - i)

    # Get input sequence
    X = data[i:i + seq_len, :]  # (seq_len, batch_size)

    # Get target sequence (shifted by 1)
    y = data[i + 1:i + 1 + seq_len, :]  # (seq_len, batch_size)
    y = y.flatten()  # (seq_len * batch_size,)

    # Convert to Tensors
    # Note: y contains integer labels but backend only supports float32
    X_tensor = Tensor(X, device=device, dtype=dtype, requires_grad=False)
    y_tensor = Tensor(y.astype(np.float32), device=device, dtype=dtype, requires_grad=False)

    return X_tensor, y_tensor

#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/18 12:06
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   PT.py
# @Desc     :   

from numpy import ndarray, random as np_random
from pandas import DataFrame, Series
from random import seed as rnd_seed, getstate, setstate
from torch import (cuda, backends, device, Tensor, tensor, float32,
                   manual_seed, get_rng_state, set_rng_state,
                   clamp, sqrt, log, nn)
from torch.utils.data import Dataset, DataLoader
from typing import Union


class TorchRandomSeed:
    """ Setting random seed for reproducibility """

    def __init__(self, description: str, seed: int = 27):
        """ Initialise the RandomSeed class
        :param description: the description of a random seed
        :param seed: the seed value to be set
        """
        self._description: str = description
        self._seed: int = seed
        self._previous_py_seed = None
        self._previous_pt_seed = None
        self._previous_np_seed = None

    def __enter__(self):
        """ Set the random seed """
        # Save the previous random seed state
        self._previous_py_seed = getstate()
        self._previous_pt_seed = get_rng_state()
        self._previous_np_seed = np_random.get_state()

        # Set the new random seed
        rnd_seed(self._seed)
        manual_seed(self._seed)
        np_random.seed(self._seed)

        print("*" * 50)
        print(f"{self._description!r} has been set randomness {self._seed}.")
        print("-" * 50)

        return self

    def __exit__(self, *args):
        """ Exit the random seed context manager """
        # Restore the previous random seed state
        if self._previous_py_seed is not None:
            setstate(self._previous_py_seed)
        if self._previous_pt_seed is not None:
            set_rng_state(self._previous_pt_seed)
        if self._previous_np_seed is not None:
            np_random.set_state(self._previous_np_seed)

        print("-" * 50)
        print(f"{self._description!r} has been restored to previous randomness.")
        print("*" * 50)
        print()

    def __repr__(self):
        """ Return a string representation of the random seed """
        return f"{self._description!r} is set to randomness {self._seed}."


def check_device() -> None:
    """ Check Available Device (CPU, GPU, MPS)
    :param option: filter option for device type
    :return: dictionary of available devices
    """

    # CUDA (NVIDIA GPU)
    if cuda.is_available():
        count: int = cuda.device_count()
        print(f"Detected {count} CUDA GPU(s):")
        for i in range(count):
            print(f"GPU {i}: {cuda.get_device_name(i)}")
            print(f"- Memory Usage:")
            print(f"- Allocated: {round(cuda.memory_allocated(i) / 1024 ** 3, 1)} GB")
            print(f"- Cached:    {round(cuda.memory_reserved(i) / 1024 ** 3, 1)} GB")

    # MPS (Apple Silicon GPU)
    elif backends.mps.is_available():
        print("Apple MPS device detected.")

    # Fallback: CPU
    else:
        print("Due to GPU or MPS unavailable, using CPU.")


def get_device(accelerator: str = "auto", cuda_mode: int = 0) -> str:
    """ Get the appropriate device based on the target device string
    :param accelerator: the target device string ("auto", "cuda", "mps", "cpu")
    :param cuda_mode: the CUDA device index to use (if applicable)
    :return: the appropriate device string
    """
    match accelerator:
        case "auto":
            if cuda.is_available():
                count: int = cuda.device_count()
                print(f"Detected {count} CUDA GPU(s):")
                if cuda_mode < count:
                    for i in range(count):
                        print(f"GPU {i}: {cuda.get_device_name(i)}")
                        print(f"- Memory Usage:")
                        print(f"- Allocated: {round(cuda.memory_allocated(i) / 1024 ** 3, 1)} GB")
                        print(f"- Cached:    {round(cuda.memory_reserved(i) / 1024 ** 3, 1)} GB")
                    print(f"The current accelerator is set to cuda:{cuda_mode}.")
                    return f"cuda:{cuda_mode}"
                else:
                    print(f"CUDA device index {cuda_mode} is out of range. Using 'cuda:0' instead.")
                    return "cuda:0"
            elif backends.mps.is_available():
                print("Apple MPS device detected.")
                return "mps"
            else:
                print("Due to GPU or MPS unavailable, using CPU ).")
                return "cpu"
        case "cuda":
            if cuda.is_available():
                count: int = cuda.device_count()
                print(f"Detected {count} CUDA GPU(s):")
                if cuda_mode < count:
                    for i in range(count):
                        print(f"GPU {i}: {cuda.get_device_name(i)}")
                        print(f"- Memory Usage:")
                        print(f"- Allocated: {round(cuda.memory_allocated(i) / 1024 ** 3, 1)} GB")
                        print(f"- Cached:    {round(cuda.memory_reserved(i) / 1024 ** 3, 1)} GB")
                    print(f"The current accelerator is set to cuda:{cuda_mode}.")
                    return f"cuda:{cuda_mode}"
                else:
                    print(f"CUDA device index {cuda_mode} is out of range. Using 'cuda:0' instead.")
                    return "cuda:0"
            else:
                print("Due to GPU unavailable, using CPU.")
                return "cpu"
        case "mps":
            if backends.mps.is_available():
                print("Apple MPS device detected.")
                return "mps"
            else:
                print("Due to MPS unavailable, using CPU.")
                return "cpu"
        case "cpu":
            print("Using CPU as target device.")
            return "cpu"

        case _:
            print("Due to GPU unavailable, using CPU.")
            return "cpu"


def arr2tensor(data: ndarray, target_device: str, is_grad: bool = False) -> Tensor:
    """ Convert a NumPy array to a PyTorch tensor
    :param data: the NumPy array to be converted
    :param target_device: the device to place the tensor on
    :param is_grad: whether the tensor requires gradient computation
    :return: the converted PyTorch tensor
    """
    return tensor(data, dtype=float32, device=target_device, requires_grad=is_grad)


def df2tensor(data: DataFrame, target_device: str, is_grad: bool = False) -> Tensor:
    """ Convert a Pandas DataFrame to a PyTorch tensor
    :param data: the DataFrame to be converted
    :param target_device: the device to place the tensor on
    :param is_grad: whether the tensor requires gradient computation
    :return: the converted PyTorch tensor
    """
    return tensor(data.values, dtype=float32, device=target_device, requires_grad=is_grad)


class TorchDataset(Dataset):
    """ A custom PyTorch Dataset class for handling features and labels """

    def __init__(self, features: Union[DataFrame, ndarray], labels: Union[DataFrame, ndarray], target_device=None):
        """ Initialise the TorchDataset class
        :param features: the feature tensor
        :param labels: the label tensor
        """
        self._features: Tensor = self._to_tensor(features, target_device)
        self._labels: Tensor = self._to_tensor(labels, target_device)

    @staticmethod
    def _to_tensor(data: Union[DataFrame, Tensor, ndarray, list], target_device=None) -> Tensor:
        """ Convert input data to a PyTorch tensor on the specified device
        :param data: the input data (DataFrame, ndarray, list, or Tensor)
        :param target_device: the target device to place the tensor on
        :return: the converted PyTorch tensor
        """
        if isinstance(data, (DataFrame, Series)):
            out = tensor(data.values, dtype=float32)
        elif isinstance(data, Tensor):
            out = data.float()
        elif isinstance(data, (ndarray, list)):
            out = tensor(data, dtype=float32)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        if target_device is None:
            target = device("cuda") if cuda.is_available() else device("cpu")
        else:
            target = target_device
        return out.to(target)

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, index: Union[int, slice]) -> Union[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """ Return a single (feature, label) pair or a batch via slice """
        if isinstance(index, slice):
            # Return a batch (for example dataset[:5])
            return self._features[index], self._labels[index]
        elif isinstance(index, int):
            # Return a single sample
            return self._features[index], self._labels[index]
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    @property
    def features(self) -> Tensor:
        return self._features

    @property
    def labels(self) -> Tensor:
        return self._labels

    def __repr__(self):
        dev = self._features.device
        return f"TorchDataset(features={self._features.shape}, labels={self._labels.shape}, device={dev})"

    def to(self, target_device: Union[str, device]) -> "TorchDataset":
        """ Move the dataset to the specified device
        :param target_device: the target device to move the dataset to
        :return: the dataset on the target device
        """
        if isinstance(target_device, str):
            target = device(target_device)
        else:
            target = target_device
        self._features = self._features.to(target)
        self._labels = self._labels.to(target)
        # Return self for chaining
        return self


class TorchDataLoader:
    """ A custom PyTorch DataLoader class for handling TorchDataset """

    def __init__(self, dataset: Dataset, batch_size: int = 32, is_shuffle: bool = True):
        """ Initialise the TorchDataLoader class
        :param dataset: the TorchDataset or Dataset to load data from
        :param batch_size: the number of samples per batch
        :param is_shuffle: whether to shuffle the data at every epoch
        """
        self._dataset: Union[Dataset, TorchDataset] = dataset
        self._batches: int = batch_size
        self._is_shuffle: bool = is_shuffle

        self._loader: DataLoader = DataLoader(
            dataset=self._dataset,
            batch_size=self._batches,
            shuffle=self._is_shuffle,
        )

    @property
    def dataset(self) -> Union[Dataset, TorchDataset]:
        return self._dataset

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """ Return a single (feature, label) pair or a batch via slice """
        if not isinstance(index, int):
            raise TypeError(f"Invalid index type: {type(index)}")
        return self._dataset[index]

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    def __repr__(self):
        return (f"TorchDataLoader(dataset={self._dataset}, "
                f"batch_size={self._batches}, "
                f"shuffle={self._is_shuffle})")


def log_mse_loss(pred: Tensor, true: Tensor, epsilon: float = 1e-7) -> Tensor:
    """ Calculate the Log Mean Squared Error Loss between predictions and true values
    - This loss is useful for regression tasks where the target values span several orders of magnitude (e.g., housing prices, population counts)
    - This loss should NOT be used when the data contains negative values
    - Predictions and targets must be strictly positive
    - The loss computes MSE in log-space, which emphasizes relative errors rather than absolute errors
    :param pred: the predicted tensor
    :param true: the true tensor
    :param epsilon: small value to avoid log(0)
    :return: the Log MSE Loss tensor
    """
    criterion = nn.MSELoss()

    pred_clamped = clamp(pred, 1, float("inf"))
    true_clamped = clamp(true, 1, float("inf"))

    pred_log = log(pred_clamped + epsilon)
    true_log = log(true_clamped + epsilon)

    return sqrt(criterion(pred_log, true_log))

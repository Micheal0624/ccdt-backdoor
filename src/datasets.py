import random
from typing import Tuple

import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, Subset


def _transforms(dataset_name: str, image_size: int):
    name = dataset_name.lower()

    if name in {"cifar10", "cifar100"}:
        train_tf = T.Compose([
            T.RandomCrop(image_size, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        eval_tf = T.Compose([
            T.ToTensor(),
        ])
        return train_tf, eval_tf

    if name == "gtsrb":
        train_tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomRotation(10),
            T.ToTensor(),
        ])
        eval_tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        return train_tf, eval_tf

    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_datasets(
    dataset_name: str,
    data_root: str,
    image_size: int = 32,
) -> Tuple[Dataset, Dataset, int]:
    name = dataset_name.lower()
    train_tf, eval_tf = _transforms(name, image_size)

    if name == "cifar10":
        train_set = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=train_tf,
        )
        test_set = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=eval_tf,
        )
        return train_set, test_set, 10

    if name == "cifar100":
        train_set = torchvision.datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=train_tf,
        )
        test_set = torchvision.datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=eval_tf,
        )
        return train_set, test_set, 100

    if name == "gtsrb":
        train_set = torchvision.datasets.GTSRB(
            root=data_root,
            split="train",
            download=True,
            transform=train_tf,
        )
        test_set = torchvision.datasets.GTSRB(
            root=data_root,
            split="test",
            download=True,
            transform=eval_tf,
        )
        return train_set, test_set, 43

    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_feedback_dataset(
    dataset_name: str,
    data_root: str,
    image_size: int = 32,
    subset_size: int = 2048,
    seed: int = 0,
) -> Dataset:
    """Create a deterministic probe subset from the training split.

    This subset uses deterministic evaluation transforms and is used by the
    feedback controller. It never draws examples from the official test set.
    The subset is sampled from the training split and does not alter the
    original training dataset, preserving comparability with the fixed-weight
    CCDT experiments.
    """

    name = dataset_name.lower()
    _, eval_tf = _transforms(name, image_size)

    if name == "cifar10":
        base = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=eval_tf,
        )
    elif name == "cifar100":
        base = torchvision.datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=eval_tf,
        )
    elif name == "gtsrb":
        base = torchvision.datasets.GTSRB(
            root=data_root,
            split="train",
            download=True,
            transform=eval_tf,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    n = len(base)
    k = min(max(1, int(subset_size)), n)
    indices = list(range(n))
    random.Random(int(seed) + 7919).shuffle(indices)
    return Subset(base, indices[:k])

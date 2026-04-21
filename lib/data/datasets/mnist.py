from typing import Optional, Dict, Iterable, Tuple, List
import torch
from torchvision import datasets
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import pandas as pd
from dataclasses import dataclass
from torchvision.utils import make_grid
from matplotlib import pyplot as plt

from lib.data import Color, colorize
from lib import filepaths


class ColoredMNISTDataset(Dataset):
    def __init__(self,
                 color_codes: Iterable[Color],
                 resize: int = 32,
                 train: bool = True,
                 random_state: int = 44
                 ):
        """
        Args:
            color_codes (Iterable[Color]): list of RGB vectors (e.g. [[1,0,0], [0,1,0], [1,1,0]])
            resize (int): resize final image tensor in shape (resize, resize)
            train (bool): whether MNIST training or test set
            random_state (int): random seed to shuffle digit_id and color_id assignments
        """
        mnist = datasets.MNIST(
            root=filepaths.ROOT_DATASETS,
            train=train,
            download=True
        )
        self.images = mnist.data / 255.
        self.labels = mnist.targets.numpy()
        self.color_codes = color_codes
        self.resize = resize

        color_ids = np.zeros_like(self.labels)
        for label in np.unique(self.labels):
            color_ids_label = [
                i * v for i, v in
                enumerate(np.array_split(
                    np.ones_like(self.labels[self.labels == label]),
                    len(color_codes)
                ))
            ]
            color_ids_label = np.concatenate(color_ids_label)
            color_ids[self.labels == label] = color_ids_label

        # dataframe aligning indexes, class label with color id
        self.data_table = pd.DataFrame(dict(
            label=self.labels,
            color_id=color_ids
        ))
        self.data_table = self.data_table.sample(frac=1, random_state=random_state).reset_index(names="idx")

    def __len__(self):
        return len(self.data_table)

    def __getitem__(self, idx):
        data_sample = self.data_table.iloc[idx]

        idx = data_sample.idx
        image = self.images[idx]
        color_id = data_sample.color_id
        rgb_code = self.color_codes[color_id]

        pixel_values = colorize(image, rgb_code).float()
        pixel_values_resized = F.resize(pixel_values, (self.resize, self.resize))
        label = torch.tensor(data_sample.label).long()
        color_id = torch.tensor(color_id).long()
        return pixel_values_resized, label, color_id


class ColoredMNISTDatasetAblated(ColoredMNISTDataset):
    def __init__(self,
                 missing_pairs: Iterable[Tuple[int, int]],
                 color_codes: Iterable[Color],
                 resize: int = 32,
                 train: bool = True,
                 random_state: int = 44
                 ):
        """
        Args:
            missing_pairs (Iterable[Tuple[int, int]]): (digit_id, color_id) pairs to omit
            color_codes (Iterable[Color]): list of RGB vectors (e.g. [[1,0,0], [0,1,0], [1,1,0]])
            resize (int): resize final image tensor in shape (resize, resize)
            train (bool): whether MNIST training or test set
            random_state (int): random seed to shuffle digit_id and color_id assignments
        """
        super().__init__(
            color_codes=color_codes,
            resize=resize,
            train=train,
            random_state=random_state
        )
        self.missing_pairs = missing_pairs

        for pairs in missing_pairs:
            filter_idx = (self.data_table.label == pairs[0]) * (self.data_table.color_id == pairs[1])
            self.data_table = self.data_table[~filter_idx].reset_index(drop=True)

    def __len__(self):
        return len(self.data_table)

    def preview_colored_digits(self, title: str = ""):
        """
        Draw one image per (color, digit) combination in a √N×√N grid,
        where N = number of valid (color, digit) pairs available.
        Assumes dataset returns (C×H×W) tensors in [0,1] plus (label, color_idx).
        """

        # Figure out which (digit_id, color_id) pairs we expect
        expected_pairs = set()
        for color_id in range(len(self.color_codes)):
            for digit_id in range(10):
                if (digit_id, color_id) in self.missing_pairs:
                    continue
                expected_pairs.add((digit_id, color_id))

        # Scan through dataset until we get one example per pair
        seen = []
        images = []

        for i in range(len(self)):
            img, digit_label, color_index = self[i]
            pair = (int(digit_label), int(color_index))
            if pair in expected_pairs and pair not in seen:
                images.append(img)
                seen.append(pair)
                if len(seen) == len(expected_pairs):
                    break

        if len(images) == 0:
            raise RuntimeError("No images found")

        # Stack into a single tensor and plot
        sorted_with_index = sorted(enumerate(seen), key=lambda x: (x[1][0], x[1][1]))
        positions = [idx for idx, _ in sorted_with_index]
        imgs_tensor = torch.stack(images)[torch.tensor(positions)]  # sort images by digit and color id ordering
        n = imgs_tensor.size(0)
        nrow = int(math.ceil(math.sqrt(n)))

        grid = make_grid(imgs_tensor, nrow=nrow, padding=2)
        plt.figure(figsize=(6, 6))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title(title)
        plt.axis('off')
        plt.show()


@dataclass
class Input:
    pixel_values: torch.FloatTensor
    label_ids: torch.LongTensor
    feature_ids: torch.LongTensor

    def to(self, device: torch.device):
        self.pixel_values = self.pixel_values.to(device)
        self.label_ids = self.label_ids.to(device)
        self.feature_ids = self.feature_ids.to(device)


def get_ColoredMNISTDataset_dataloader(
        batch_size: int,
        dataset: Dataset,
        **kwargs
):
    """
    Args:
        batch_size (int): batch size for the dataloaders
        dataset (Dataset): Dataset instances of Colored MNIST
    """

    def collate_fn(batch):
        pixel_values = []
        label_ids = []
        feature_ids = []
        for img, label_id, feature_id in batch:
            pixel_values.append(img)
            label_ids.append(label_id)
            feature_ids.append(feature_id)

        return Input(
            pixel_values=torch.stack(pixel_values),
            label_ids=torch.stack(label_ids),
            feature_ids=torch.stack(feature_ids)
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        **kwargs
    )

    return dataloader


def get_ColoredMNISTDataset_train_test_dataloaders(
        batch_size: int,
        train_dataset: Dataset,
        test_dataset: Dataset,
        **kwargs
):
    """
    Args:
        batch_size (int): batch size for the dataloaders
        train_dataset/test_dataset (Dataset): Dataset instances of Colored MNIST
    """

    def collate_fn(batch):
        pixel_values = []
        label_ids = []
        feature_ids = []
        for img, label_id, feature_id in batch:
            pixel_values.append(img)
            label_ids.append(label_id)
            feature_ids.append(feature_id)

        return Input(
            pixel_values=torch.stack(pixel_values),
            label_ids=torch.stack(label_ids),
            feature_ids=torch.stack(feature_ids)
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn,
        **kwargs
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn,
        **kwargs
    )

    return train_dataloader, test_dataloader

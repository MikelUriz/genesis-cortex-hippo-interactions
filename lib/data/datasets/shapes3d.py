from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader



class HueColor(IntEnum):
    RED = 0
    ORANGE = 1
    YELLOW_GREEN = 2
    GREEN = 3
    TURQUOISE = 4
    CYAN = 5
    BLUE = 6
    VIOLET = 7
    MAGENTA = 8
    PINK = 9


class Shape(IntEnum):
    CUBE = 0
    CYLINDER = 1
    SPHERE = 2
    PILL = 3


class Scale(IntEnum):
    TINY = 0      # 0.75
    VERY_SMALL = 1
    SMALL = 2
    MEDIUM_SMALL = 3
    MEDIUM_LARGE = 4
    LARGE = 5
    VERY_LARGE = 6
    HUGE = 7      # 1.25


@dataclass
class Shapes3DFactors:
    floor_hue: int | torch.Tensor | None = None
    wall_hue: int | torch.Tensor | None = None
    obj_hue: int | torch.Tensor | None = None
    scale: int | torch.Tensor | None = None
    shape: int | torch.Tensor | None = None
    orientation: int | torch.Tensor | None = None

    def to(self, device: torch.device):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                setattr(self, field.name, value.to(device))
        return self


class Shapes3DDataset(Dataset):
    FACTOR_MAP = {
        'floor_hue': 0, 'wall_hue': 1, 'obj_hue': 2,
        'scale': 3, 'shape': 4, 'orientation': 5
    }

    def __init__(self,
                 file_path: str,
                 active_features: list[str] | None = None,
                 train: bool = True,
                 split_frac: float = 0.8,
                 random_state: int = 44,
                 resize: int | None = None):
        """
        Args:
            file_path: Path to 3dshapes.h5
            active_features: List of feature names to return
            train: If True, returns training set, else test set
            split_frac: Fraction of data for training
            random_state: Seed for reproducibility and splitting
            resize: Optional image resize
        """
        self.file_path = file_path
        self.active_features = active_features
        self.resize = resize

        # Validate features
        if active_features is None:
            active_features = ['floor_hue', 'wall_hue', 'obj_hue', 'scale', 'shape', 'orientation']

        for feat in active_features:
            if feat not in self.FACTOR_MAP:
                raise ValueError(f"Feature '{feat}' not recognized. Use: {list(self.FACTOR_MAP.keys())}")

        # Load labels and create discrete indices
        with h5py.File(self.file_path, 'r') as f:
            raw_labels = f['labels'][:]
            self.num_total = raw_labels.shape[0]
            self.images = f['images'][:]

        # Create a metadata table
        df_dict = {'h5_idx': np.arange(self.num_total)}
        for name, col_idx in self.FACTOR_MAP.items():
            unique_vals = np.unique(raw_labels[:, col_idx])
            val_to_idx = {val: i for i, val in enumerate(unique_vals)}
            df_dict[name] = [val_to_idx[v] for v in raw_labels[:, col_idx]]

        full_df = pd.DataFrame(df_dict)

        # Stratified Split
        # To stratify across multiple features, we create a combined key
        # We use all features to define the strata
        stratify_key = full_df[active_features].apply(lambda x: '_'.join(x.values.astype(str)), axis=1)

        train_df, test_df = train_test_split(
            full_df,
            train_size=split_frac,
            stratify=stratify_key,
            random_state=random_state
        )

        self.data_table = train_df.reset_index(drop=True) if train else test_df.reset_index(drop=True)

    def __len__(self):
        return len(self.data_table)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Shapes3DFactors]:

        sample = self.data_table.iloc[idx]

        # Load Image
        img = self.images[int(sample.h5_idx)]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.resize:
            img = F.resize(img, (self.resize, self.resize))

        # Build Factors Dataclass
        feature_kwargs = {}
        for feat in self.active_features:
            feature_kwargs[feat] = int(sample[feat])

        factors = Shapes3DFactors(**feature_kwargs)

        return img, factors

    def get_factor_sizes(self) -> dict:
        """Returns a dictionary mapping factor names to the count of their unique discrete values."""
        return {name: int(self.data_table[name].nunique()) for name in self.FACTOR_MAP.keys()}


def get_Shapes3D_datasets(
        file_path: str,
        active_features: list[str] | None = None,
        split_frac: float = 0.8,
        random_state: int = 44,
        resize: int | None = None
) -> Tuple[Shapes3DDataset, Shapes3DDataset]:
        """
        Args:
            file_path: Path to 3dshapes.h5
            active_features: List of feature names to return
            split_frac: Fraction of data for training
            random_state: Seed for reproducibility and splitting
            resize: Optional image resize
        """
        train_ds = Shapes3DDataset(
            file_path=file_path,
            active_features=active_features,
            train=True,
            split_frac=split_frac,
            random_state=random_state,
            resize=resize,
        )

        test_ds = Shapes3DDataset(
            file_path=file_path,
            active_features=active_features,
            train=False,
            split_frac=split_frac,
            random_state=random_state,
            resize=resize,
        )

        return train_ds, test_ds


@dataclass
class Input:
    pixel_values: torch.FloatTensor
    factors: Shapes3DFactors

    def to(self, device: torch.device):
        self.pixel_values = self.pixel_values.to(device)
        self.factors = self.factors.to(device)
        return self


def shapes_collate_fn(batch):
    imgs, factors_list = zip(*batch)

    batch_kwargs = {}
    for field in fields(Shapes3DFactors):
        val = getattr(factors_list[0], field.name)
        if val is not None:
            batch_kwargs[field.name] = torch.tensor([getattr(f, field.name) for f in factors_list])
        else:
            batch_kwargs[field.name] = None

    return Input(
        pixel_values=torch.stack(imgs),
        factors=Shapes3DFactors(**batch_kwargs)
    )


def get_Shapes3D_dataloaders(
        batch_size: int,
        train_dataset: Shapes3DDataset,
        test_dataset: Shapes3DDataset,
        **kwargs
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=shapes_collate_fn,
        **kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=shapes_collate_fn,
        **kwargs
    )
    return train_loader, test_loader

from dataclasses import dataclass
from typing import Tuple

import medmnist
from medmnist import INFO

from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


@dataclass
class DataSpec:
    n_classes: int
    task: str
    channels: int
    img_size: int


def get_medmnist_spec(dataset_name: str = "dermamnist") -> DataSpec:
    info = INFO[dataset_name]
    label = info.get("label", {})

    if isinstance(label, dict):
        n_classes = len(label)
    elif isinstance(label, (list, tuple)):
        n_classes = len(label)
    else:
        n_classes = int(info.get("n_labels", 0))

    return DataSpec(
        n_classes=n_classes,
        task=str(info.get("task", "multi-class")),
        channels=int(info.get("n_channels", 3)),
        img_size=28,
    )


def get_dermamnist_spec() -> DataSpec:
    return get_medmnist_spec("dermamnist")


def build_transforms(train: bool, channels: int) -> transforms.Compose:
    normalize_channels = 3 if channels == 1 else channels
    mean = [0.5] * normalize_channels
    std = [0.5] * normalize_channels

    common = [
        transforms.ToTensor(),
    ]

    if channels == 1:
        common.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

    if train:
        return transforms.Compose([
            *common,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.Normalize(mean=mean, std=std),
        ])

    return transforms.Compose([
        *common,
        transforms.Normalize(mean=mean, std=std),
    ])


def _get_data_class(dataset_name: str):
    data_class_name = INFO[dataset_name]["python_class"]
    return getattr(medmnist, data_class_name)


def _extract_labels(ds):
    labels = ds.labels
    return labels.squeeze().tolist()


def get_dataloaders(
    root: str = "data/medmnist",
    batch_size: int = 128,
    num_workers: int = 2,
    dataset_name: str = "dermamnist",
) -> Tuple[DataLoader, DataLoader, DataLoader, DataSpec]:
    spec = get_medmnist_spec(dataset_name)
    DataClass = _get_data_class(dataset_name)

    train_ds = DataClass(
        split="train",
        root=root,
        transform=build_transforms(True, spec.channels),
        download=True,
    )

    if hasattr(get_dataloaders, "subset_config"):
        subset_cfg = get_dataloaders.subset_config

        if subset_cfg.get("use_subset", False):
            import json
            from pathlib import Path

            subset_path = subset_cfg.get("subset_indices_path")
            if subset_path is None:
                raise ValueError("subset_indices_path must be provided when use_subset=True")

            subset_path = Path(subset_path)
            if not subset_path.exists():
                raise FileNotFoundError(f"Subset file not found: {subset_path}")

            print(f"[INFO] Loading subset indices from: {subset_path}")

            with open(subset_path, "r", encoding="utf-8") as f:
                subset_data = json.load(f)

            subset_indices = subset_data["indices"]

            print(f"[INFO] Using subset: {len(subset_indices)} samples")
            train_ds = Subset(train_ds, subset_indices)

    val_ds = DataClass(
        split="val",
        root=root,
        transform=build_transforms(False, spec.channels),
        download=True,
    )
    test_ds = DataClass(
        split="test",
        root=root,
        transform=build_transforms(False, spec.channels),
        download=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, spec


def get_kfold_dataloaders(
    root: str = "data/medmnist",
    batch_size: int = 128,
    num_workers: int = 2,
    fold_index: int = 0,
    n_splits: int = 5,
    seed: int = 42,
    train_subset_fraction: float | None = None,
    train_subset_seed: int | None = None,
    dataset_name: str = "dermamnist",
) -> Tuple[DataLoader, DataLoader, DataLoader, DataSpec]:
    spec = get_medmnist_spec(dataset_name)
    DataClass = _get_data_class(dataset_name)

    train_train_ds = DataClass(
        split="train",
        root=root,
        transform=build_transforms(True, spec.channels),
        download=True,
    )
    val_train_ds = DataClass(
        split="val",
        root=root,
        transform=build_transforms(True, spec.channels),
        download=True,
    )
    dev_train_ds = ConcatDataset([train_train_ds, val_train_ds])

    train_eval_ds = DataClass(
        split="train",
        root=root,
        transform=build_transforms(False, spec.channels),
        download=True,
    )
    val_eval_ds = DataClass(
        split="val",
        root=root,
        transform=build_transforms(False, spec.channels),
        download=True,
    )
    dev_eval_ds = ConcatDataset([train_eval_ds, val_eval_ds])

    test_ds = DataClass(
        split="test",
        root=root,
        transform=build_transforms(False, spec.channels),
        download=True,
    )

    labels = _extract_labels(train_eval_ds) + _extract_labels(val_eval_ds)
    indices = list(range(len(labels)))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(indices, labels))

    if fold_index < 0 or fold_index >= n_splits:
        raise ValueError(f"fold_index must be in [0, {n_splits - 1}], got {fold_index}")

    train_idx, val_idx = folds[fold_index]
    train_idx = train_idx.tolist()
    val_idx = val_idx.tolist()

    if train_subset_fraction is not None:
        if not (0.0 < float(train_subset_fraction) <= 1.0):
            raise ValueError(f"train_subset_fraction must be in (0, 1], got {train_subset_fraction}")

        subset_seed = seed if train_subset_seed is None else int(train_subset_seed)

        fold_train_labels = [labels[i] for i in train_idx]
        fold_local_indices = list(range(len(train_idx)))

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=float(train_subset_fraction),
            random_state=subset_seed,
        )

        selected_local_idx, _ = next(splitter.split(fold_local_indices, fold_train_labels))
        selected_local_idx = sorted(selected_local_idx.tolist())

        original_fold_train_size = len(train_idx)
        train_idx = [train_idx[i] for i in selected_local_idx]

        print(
            f"[INFO] Using k-fold train subset | "
            f"fraction={train_subset_fraction} | "
            f"selected={len(train_idx)} / original_fold_train={original_fold_train_size} | "
            f"subset_seed={subset_seed}"
        )

    train_subset = Subset(dev_train_ds, train_idx)
    val_subset = Subset(dev_eval_ds, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, spec

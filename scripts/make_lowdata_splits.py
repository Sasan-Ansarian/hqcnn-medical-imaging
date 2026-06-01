import json
from pathlib import Path

import medmnist
from medmnist import INFO
from sklearn.model_selection import StratifiedShuffleSplit


def get_dermamnist_train_labels(root: str):
    data_class_name = INFO["dermamnist"]["python_class"]
    DataClass = getattr(medmnist, data_class_name)

    train_ds = DataClass(split="train", root=root, download=True)
    labels = train_ds.labels.squeeze().tolist()
    return labels


def class_counts_from_indices(labels, indices):
    counts = {}
    for idx in indices:
        y = int(labels[idx])
        counts[y] = counts.get(y, 0) + 1
    return dict(sorted(counts.items()))


def make_split(labels, fraction: float, seed: int):
    indices = list(range(len(labels)))

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=fraction,
        random_state=seed
    )

    selected_idx, _ = next(splitter.split(indices, labels))
    selected_idx = sorted(selected_idx.tolist())
    return selected_idx


def save_split(root: str, out_path: str, fraction: float, seed: int):
    labels = get_dermamnist_train_labels(root)
    selected_idx = make_split(labels, fraction, seed)
    counts = class_counts_from_indices(labels, selected_idx)

    payload = {
        "dataset": "dermamnist",
        "split": "train",
        "fraction": fraction,
        "seed": seed,
        "stratified": True,
        "num_selected": len(selected_idx),
        "indices": selected_idx,
        "class_counts": counts,
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[OK] wrote {out_file}")
    print(f"[INFO] num_selected={len(selected_idx)}")
    print(f"[INFO] class_counts={counts}")


def main():
    project_root = Path("/scratch/lustre/home/saan1491/hqcnn_project")
    data_root = str(project_root / "data")
    split_dir = project_root / "splits" / "chapter7" / "lowdata"

    configs = [
        (0.10, 42, split_dir / "split_10_seed42.json"),
        (0.25, 42, split_dir / "split_25_seed42.json"),
        (0.50, 42, split_dir / "split_50_seed42.json"),
    ]

    for fraction, seed, out_path in configs:
        save_split(
            root=data_root,
            out_path=str(out_path),
            fraction=fraction,
            seed=seed,
        )


if __name__ == "__main__":
    main()

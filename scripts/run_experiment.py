from src.models.factory import build_model
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.utils.seed import set_seed
from src.data.dermamnist import get_dataloaders, get_kfold_dataloaders
from src.train.engine import train_one_epoch, evaluate


def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pretrained_backbone(model, pretrained_path: str, device: torch.device):
    print(f"[INFO] Loading pretrained backbone from: {pretrained_path}")
    ckpt = torch.load(pretrained_path, map_location=device)
    state_dict = ckpt["model_state"]

    backbone_state = {}
    for k, v in state_dict.items():
        # New checkpoint format: keys already stored under backbone.features.*
        if k.startswith("backbone.features."):
            new_k = k.replace("backbone.", "", 1)
            backbone_state[new_k] = v
            continue

        # Ignore classifier/head keys from wrapper checkpoints
        if k.startswith("head.") or k.startswith("fc.") or k.startswith("classifier."):
            continue

        # Old torchvision-style checkpoint format
        if k.startswith("conv1."):
            new_k = k.replace("conv1", "features.0", 1)
        elif k.startswith("bn1."):
            new_k = k.replace("bn1", "features.1", 1)
        elif k.startswith("layer1."):
            new_k = k.replace("layer1", "features.4", 1)
        elif k.startswith("layer2."):
            new_k = k.replace("layer2", "features.5", 1)
        elif k.startswith("layer3."):
            new_k = k.replace("layer3", "features.6", 1)
        elif k.startswith("layer4."):
            new_k = k.replace("layer4", "features.7", 1)
        else:
            continue

        backbone_state[new_k] = v

    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
    print(f"[INFO] Loaded backbone weights: {len(backbone_state)} tensors")
    print(f"[INFO] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")


def build_optimizer(model, training_cfg: dict):
    opt_name = training_cfg.get("optimizer", "adamw").lower()
    weight_decay = float(training_cfg.get("weight_decay", 0.0))

    backbone_lr = training_cfg.get("backbone_lr", None)
    head_lr = training_cfg.get("head_lr", None)

    if backbone_lr is None or head_lr is None:
        lr = float(training_cfg["lr"])
        params = [p for p in model.parameters() if p.requires_grad]
        if opt_name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    backbone_params = model.get_backbone_params()
    head_params = model.get_head_params()

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": float(backbone_lr)})
    if head_params:
        param_groups.append({"params": head_params, "lr": float(head_lr)})

    if opt_name == "adam":
        return torch.optim.Adam(param_groups, weight_decay=weight_decay)
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def get_selection_metric(eval_stats, selection_metric: str) -> float:
    if selection_metric == "val_acc":
        return float(eval_stats.accuracy)
    if selection_metric == "val_macro_f1":
        return float(eval_stats.macro_f1)
    if selection_metric == "val_balanced_accuracy":
        return float(eval_stats.balanced_accuracy)
    raise ValueError(f"Unknown selection metric: {selection_metric}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    exp = cfg["experiment"]
    training = cfg["training"]
    dataset = cfg["dataset"]
    model_cfg = cfg["model"]

    set_seed(int(exp["seed"]))

    out_dir = Path(exp["output_dir"]) / exp["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = bool(training.get("device", "cuda") == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("[INFO] device:", device)

    Path(dataset["root"]).mkdir(parents=True, exist_ok=True)

    dataset_name = str(dataset.get("name", "dermamnist")).lower()
    print(f"[INFO] dataset.name={dataset_name}")

    use_kfold = bool(dataset.get("use_kfold", False))

    if use_kfold:
        fold_index = int(dataset["fold_index"])
        n_splits = int(dataset.get("n_splits", 5))

        print(f"[INFO] Using k-fold CV | fold={fold_index} | n_splits={n_splits}")

        train_loader, val_loader, test_loader, spec = get_kfold_dataloaders(
            root=dataset["root"],
            batch_size=int(training["batch_size"]),
            num_workers=int(training["num_workers"]),
            fold_index=fold_index,
            n_splits=n_splits,
            seed=int(exp["seed"]),
            train_subset_fraction=dataset.get("kfold_train_subset_fraction"),
            train_subset_seed=dataset.get("kfold_train_subset_seed"),
            dataset_name=dataset_name,
        )
    else:
        get_dataloaders.subset_config = {
            "use_subset": dataset.get("use_subset", False),
            "subset_indices_path": dataset.get("subset_indices_path"),
        }
        train_loader, val_loader, test_loader, spec = get_dataloaders(
            root=dataset["root"],
            batch_size=int(training["batch_size"]),
            num_workers=int(training["num_workers"]),
            dataset_name=dataset_name,
        )

    model = build_model(model_cfg=model_cfg, num_classes=int(spec.n_classes)).to(device)
    model_name = model_cfg.get("name", "unknown")

    pretrained_path = model_cfg.get("pretrained_backbone")
    if pretrained_path is not None:
        load_pretrained_backbone(model, pretrained_path, device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] model.name={model_name} | model_class={model.__class__.__name__} | params={n_params:,} | trainable={n_trainable:,}")

    class_weights = None
    loss_cfg = training.get("loss", {})

    if loss_cfg.get("use_class_weights", False):
        weight_values = loss_cfg.get("class_weights")
        if weight_values is None:
            raise ValueError("use_class_weights=True but class_weights is missing in config")
        class_weights = torch.tensor(weight_values, dtype=torch.float32, device=device)
        print(f"[INFO] Using class-weighted CE: {weight_values}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    freeze_backbone = bool(training.get("freeze_backbone", False))
    use_warmup = bool(training.get("use_warmup", False))
    warmup_epochs = int(training.get("warmup_epochs", 0))
    selection_metric = str(training.get("selection_metric", "val_acc"))

    if freeze_backbone and use_warmup:
        print("[WARN] freeze_backbone=True overrides warm-up. Disabling warm-up.")
        use_warmup = False
        warmup_epochs = 0

    if freeze_backbone:
        print("[INFO] Frozen-backbone screening enabled")
        model.freeze_backbone()
    elif use_warmup:
        print(f"[INFO] Warm-up enabled for {warmup_epochs} epochs")
        model.freeze_backbone()

    optimizer = build_optimizer(model, training)

    n_trainable_after_setup = sum(p.numel() for p in model.parameters() if p.requires_grad)

    history = {
        "model_name": model_name,
        "num_params": n_params,
        "num_trainable_params_initial": n_trainable,
        "num_trainable_params_after_setup": n_trainable_after_setup,
        "freeze_backbone": freeze_backbone,
        "selection_metric": selection_metric,
        "epochs": [],
    }

    best_score = -1.0
    best_path = out_dir / "best_model.pt"
    warmup_finished = False

    t0 = time.time()
    for epoch in range(int(training["epochs"])):
        if use_warmup and (not warmup_finished) and epoch == warmup_epochs:
            print(f"[INFO] Warm-up finished at epoch {epoch}. Unfreezing backbone.")
            model.unfreeze_backbone()
            optimizer = build_optimizer(model, training)
            warmup_finished = True

            n_trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[INFO] Trainable params after unfreeze: {n_trainable_now:,}")

        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch + 1,
            "train_loss": tr.loss,
            "train_acc": tr.accuracy,
            "grad_norm_mean": tr.grad_norm_mean,
            "grad_norm_var": tr.grad_norm_var,
            "val_loss": va.loss,
            "val_acc": va.accuracy,
            "val_macro_f1": va.macro_f1,
            "val_balanced_accuracy": va.balanced_accuracy,
        }
        print(row)
        history["epochs"].append(row)

        score = get_selection_metric(va, selection_metric)
        if score > best_score:
            best_score = score
            torch.save({"model_state": model.state_dict(), "cfg": cfg}, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    te = evaluate(model, test_loader, criterion, device)

    summary = {
        "best_selection_score": float(best_score),
        "selection_metric": selection_metric,
        "test_loss": float(te.loss),
        "test_acc": float(te.accuracy),
        "test_macro_f1": float(te.macro_f1),
        "test_balanced_accuracy": float(te.balanced_accuracy),
        "test_conf_mat": te.conf_mat.tolist(),
        "runtime_sec": float(time.time() - t0),
        "freeze_backbone": freeze_backbone,
        "use_warmup": use_warmup,
        "warmup_epochs": warmup_epochs,
        "backbone_lr": training.get("backbone_lr"),
        "head_lr": training.get("head_lr"),
        "global_lr": training.get("lr"),
    }

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[INFO] Saved:", out_dir)
    print("[INFO] Summary:", summary)


if __name__ == "__main__":
    main()

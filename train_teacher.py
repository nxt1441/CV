"""Week 1 — Fine-tune ResNet-50 teacher on CUB-200-2011.

Usage:
    python train_teacher.py
"""

import random
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from dataset import get_loaders
from model import load_teacher


# ── reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── accuracy helpers ─────────────────────────────────────────────────────────

def topk_accuracy(logits, labels, k=5):
    with torch.no_grad():
        pred = logits.topk(k, dim=1).indices          # [B, k]
        correct = pred.eq(labels.view(-1, 1).expand_as(pred))
        return correct.any(1).float().mean().item() * 100


# ── train / eval loops ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, top1_sum, n = 0.0, 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        top1_sum += (logits.argmax(1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, top1_sum / n * 100


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    top1_sum, top5_sum, n = 0.0, 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        top1_sum += (logits.argmax(1) == labels).sum().item()
        top5_sum += topk_accuracy(logits, labels, k=5) / 100 * labels.size(0)
        n += labels.size(0)
    return top1_sum / n * 100, top5_sum / n * 100


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = cfg["dataset"]
    tc = cfg["teacher"]

    train_loader, val_loader = get_loaders(
        root=ds["root"],
        img_size=ds["img_size"],
        batch_size=ds["batch_size"],
        num_workers=ds["num_workers"],
    )

    model = load_teacher(num_classes=ds["num_classes"],
                         weights_path=tc["weights_path"]).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Teacher params: {params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=tc["lr"],
                                  weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=tc["epochs"])
    criterion = nn.CrossEntropyLoss(label_smoothing=tc["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda")

    Path("results").mkdir(exist_ok=True)
    log_path = "results/teacher_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_top1", "val_top1", "val_top5"])

    patience = tc.get("patience", 0)   # 0 disables early stopping
    best_top1 = 0.0
    epochs_no_improve = 0
    for epoch in range(1, tc["epochs"] + 1):
        train_loss, train_top1 = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_top1, val_top5 = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"[{epoch:3d}/{tc['epochs']}]  "
              f"loss={train_loss:.4f}  train={train_top1:.2f}%  "
              f"val_top1={val_top1:.2f}%  val_top5={val_top5:.2f}%")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.4f}", f"{train_top1:.2f}",
                 f"{val_top1:.2f}", f"{val_top5:.2f}"]
            )

        if val_top1 > best_top1:
            best_top1 = val_top1
            epochs_no_improve = 0
            torch.save(model.state_dict(), tc["checkpoint"])
            print(f"  -> Saved best model (val_top1={best_top1:.2f}%)")
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                print(f"  -> Early stopping: no val_top1 improvement for {patience} epochs")
                break

    print(f"\nDone. Best val top-1: {best_top1:.2f}%")


if __name__ == "__main__":
    main()

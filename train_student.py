"""Weeks 2–5 — Train student (MobileNetV2) with various methods.

Week 2:  python train_student.py --method baseline_a
         python train_student.py --method baseline_b

Week 3:  python train_student.py --method hinton
         python train_student.py --method hinton --temperature 2
         python train_student.py --method hinton --temperature 8

Week 4:  python train_student.py --method feature
         python train_student.py --method attention

Week 5:  python train_student.py --method rkd
         python train_student.py --method combined
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from dataset import get_loaders
from model import load_teacher
from student_model import load_student
from distillation import (
    HintonKDLoss, FeatureKDLoss, AttentionTransferLoss, RKDLoss, CombinedKDLoss
)

METHODS = ("baseline_a", "baseline_b", "hinton", "feature", "attention", "rkd", "combined")

# MobileNetV2 features[13] → 96ch, 14×14  |  ResNet-50 layer3 → 1024ch, 14×14
# features[14] outputs 160ch at 7×7 (after stride-2 block) — wrong spatial size
STUDENT_FEAT_CHANNELS = 96
TEACHER_FEAT_CHANNELS = 1024


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def topk_accuracy(logits, labels, k=5):
    with torch.no_grad():
        pred = logits.topk(k, dim=1).indices
        correct = pred.eq(labels.view(-1, 1).expand_as(pred))
        return correct.any(1).float().mean().item() * 100


def register_hooks(teacher, student):
    """Attach forward hooks to extract intermediate features and embeddings."""
    cache = {}

    def hook(name):
        def fn(_, __, output):
            cache[name] = output
        return fn

    teacher.layer3.register_forward_hook(hook("t_feat"))       # [B, 1024, 14, 14]
    student.features[13].register_forward_hook(hook("s_feat")) # [B,   96, 14, 14]

    # Embeddings: global-average-pooled final features (before classifier)
    teacher.avgpool.register_forward_hook(hook("t_emb"))
    student.features[-1].register_forward_hook(hook("s_emb"))

    return cache


def train_epoch(student, teacher, loader, optimizer, criterion, scaler, device,
                method, cache):
    student.train()
    if teacher:
        teacher.eval()

    total_loss, top1_sum, n = 0.0, 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            if teacher:
                with torch.no_grad():
                    t_logits = teacher(images)
            s_logits = student(images)

            if method in ("baseline_a", "baseline_b"):
                loss = criterion(s_logits, labels)

            elif method == "hinton":
                loss = criterion(s_logits, t_logits, labels)

            elif method in ("feature", "attention"):
                loss = criterion(s_logits, t_logits, labels,
                                 cache["s_feat"], cache["t_feat"])

            elif method == "rkd":
                # avg-pool spatial dims before flatten: [B,C,H,W] → [B,C]
                s_emb = F.adaptive_avg_pool2d(cache["s_emb"], 1).flatten(1)
                t_emb = F.adaptive_avg_pool2d(cache["t_emb"], 1).flatten(1)
                loss = criterion(s_logits, t_logits, labels, s_emb, t_emb)

            elif method == "combined":
                loss = criterion(s_logits, t_logits, labels,
                                 cache["s_feat"], cache["t_feat"])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        top1_sum += (s_logits.argmax(1) == labels).sum().item()
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature for hinton (for sweep)")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = get_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Method: {args.method} | Device: {device}")

    ds = cfg["dataset"]
    sc = cfg["student"]
    kd = cfg["distillation"]

    train_loader, val_loader = get_loaders(
        root=ds["root"], img_size=ds["img_size"],
        batch_size=ds["batch_size"], num_workers=ds["num_workers"],
    )

    # ── teacher (frozen) ──────────────────────────────────────────────────────
    teacher = None
    cache = {}
    if args.method not in ("baseline_a", "baseline_b"):
        teacher = load_teacher(
            num_classes=ds["num_classes"],
            weights_path=cfg["teacher"]["checkpoint"],   # fine-tuned teacher
        ).to(device)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()

    # ── student ───────────────────────────────────────────────────────────────
    weights = sc["weights_path"] if args.method != "baseline_a" else None
    student = load_student(num_classes=ds["num_classes"], weights_path=weights).to(device)

    if teacher:
        cache = register_hooks(teacher, student)

    # ── loss ──────────────────────────────────────────────────────────────────
    T     = args.temperature if args.temperature else kd["temperature"]
    alpha = args.alpha if args.alpha else kd["alpha"]
    beta  = kd["beta"]

    if args.method in ("baseline_a", "baseline_b"):
        criterion = nn.CrossEntropyLoss()
    elif args.method == "hinton":
        criterion = HintonKDLoss(temperature=T, alpha=alpha)
    elif args.method == "feature":
        criterion = FeatureKDLoss(STUDENT_FEAT_CHANNELS, TEACHER_FEAT_CHANNELS, beta=beta)
    elif args.method == "attention":
        criterion = AttentionTransferLoss(beta=beta)
    elif args.method == "rkd":
        criterion = RKDLoss(lambda_d=kd["lambda_d"], lambda_a=kd["lambda_a"])
    elif args.method == "combined":
        criterion = CombinedKDLoss(STUDENT_FEAT_CHANNELS, TEACHER_FEAT_CHANNELS,
                                   temperature=T, alpha=alpha, beta=beta)

    criterion = criterion.to(device)

    # ── optimiser ─────────────────────────────────────────────────────────────
    all_params = list(student.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=sc["lr"], weight_decay=sc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sc["epochs"])
    scaler = torch.amp.GradScaler("cuda")

    # ── logging ───────────────────────────────────────────────────────────────
    run_name = args.method
    if args.temperature:
        run_name += f"_T{int(args.temperature)}"

    Path("results").mkdir(exist_ok=True)
    log_path = f"results/{run_name}_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_top1", "val_top1", "val_top5"])

    # ── training loop ─────────────────────────────────────────────────────────
    patience = sc.get("patience", 0)   # 0 disables early stopping
    best_top1 = 0.0
    epochs_no_improve = 0
    for epoch in range(1, sc["epochs"] + 1):
        train_loss, train_top1 = train_epoch(
            student, teacher, train_loader, optimizer, criterion,
            scaler, device, args.method, cache,
        )
        val_top1, val_top5 = evaluate(student, val_loader, device)
        scheduler.step()

        print(f"[{epoch:3d}/{sc['epochs']}]  "
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
            ckpt = f"results/{run_name}_best.pth"
            torch.save(student.state_dict(), ckpt)
            print(f"  -> Saved best ({best_top1:.2f}%)")
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                print(f"  -> Early stopping: no val_top1 improvement for {patience} epochs")
                break

    print(f"\nDone. Best val top-1: {best_top1:.2f}%")


if __name__ == "__main__":
    main()

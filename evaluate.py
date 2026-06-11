"""Week 5 — Evaluate all saved checkpoints and print a results table.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint results/hinton_best.pth
"""

import argparse
import time
import csv
from pathlib import Path

import torch
import torch.nn as nn

from dataset import get_loaders
from model import load_teacher
from student_model import load_student
import yaml


def topk_accuracy(logits, labels, k=5):
    with torch.no_grad():
        pred = logits.topk(k, dim=1).indices
        correct = pred.eq(labels.view(-1, 1).expand_as(pred))
        return correct.any(1).float().mean().item() * 100


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    top1_sum, top5_sum, n = 0.0, 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        top1_sum += (logits.argmax(1) == labels).sum().item()
        top5_sum += topk_accuracy(logits, labels, k=5) / 100 * labels.size(0)
        n += labels.size(0)
    return top1_sum / n * 100, top5_sum / n * 100


def measure_latency(model, device, img_size=224, n_runs=200, warmup=20):
    model.eval()
    x = torch.randn(1, 3, img_size, img_size).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        start = time.perf_counter()
        for _ in range(n_runs):
            model(x)
    return (time.perf_counter() - start) / n_runs * 1000


def model_stats(model):
    params = sum(p.numel() for p in model.parameters())
    size_mb = params * 4 / 1024 ** 2
    return params, size_mb


def evaluate_checkpoint(name, model, loader, device, img_size):
    top1, top5 = eval_model(model, loader, device)
    latency = measure_latency(model, device, img_size)
    params, size_mb = model_stats(model)
    return {
        "name": name,
        "top1": f"{top1:.2f}",
        "top5": f"{top5:.2f}",
        "params_M": f"{params/1e6:.2f}",
        "size_mb": f"{size_mb:.1f}",
        "latency_ms": f"{latency:.2f}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Evaluate a single checkpoint instead of all")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = cfg["dataset"]

    _, val_loader = get_loaders(
        root=ds["root"], img_size=ds["img_size"],
        batch_size=ds["batch_size"], num_workers=ds["num_workers"],
    )

    # Map checkpoint names to (model_type, path)
    if args.checkpoint:
        checkpoints = [("custom", "student", args.checkpoint)]
    else:
        checkpoints = [
            ("Teacher (ResNet-50)",    "teacher",  cfg["teacher"]["checkpoint"]),
            ("Baseline-A (scratch)",   "student",  "results/baseline_a_best.pth"),
            ("Baseline-B (pretrain)",  "student",  "results/baseline_b_best.pth"),
            ("Hinton KD (T=2)",        "student",  "results/hinton_T2_best.pth"),
            ("Hinton KD (T=4)",        "student",  "results/hinton_best.pth"),
            ("Hinton KD (T=8)",        "student",  "results/hinton_T8_best.pth"),
            ("Feature KD",             "student",  "results/feature_best.pth"),
            ("Attention Transfer",     "student",  "results/attention_best.pth"),
            ("RKD",                    "student",  "results/rkd_best.pth"),
            ("Combined KD",            "student",  "results/combined_best.pth"),
        ]

    rows = []
    header = f"{'Method':<25} {'Top-1':>6} {'Top-5':>6} {'Params(M)':>10} {'Size(MB)':>9} {'Lat(ms)':>8}"
    print("\n" + header)
    print("-" * len(header))

    for name, mtype, ckpt_path in checkpoints:
        if not Path(ckpt_path).exists():
            print(f"  {name:<23} — checkpoint not found, skipping")
            continue

        if mtype == "teacher":
            model = load_teacher(num_classes=ds["num_classes"],
                                  weights_path=ckpt_path).to(device)
        else:
            model = load_student(num_classes=ds["num_classes"]).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

        r = evaluate_checkpoint(name, model, val_loader, device, ds["img_size"])
        rows.append(r)
        print(f"  {r['name']:<23} {r['top1']:>6} {r['top5']:>6} "
              f"{r['params_M']:>10} {r['size_mb']:>9} {r['latency_ms']:>8}")

    # Save to CSV
    if rows:
        out = "results/final_results.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

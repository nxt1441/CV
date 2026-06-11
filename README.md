# Knowledge Distillation: Compressing Large CV Models
### Complete Reproduction Guide — Weeks 1 to 5

---

## What This Project Does

This project trains a large **ResNet-50 teacher** model on a fine-grained bird classification dataset (CUB-200-2011, 200 species), then compresses its knowledge into a much smaller **MobileNetV2 student** using three distillation strategies:

- **Response-Based KD** (Hinton et al., 2015) — match teacher's soft output probabilities
- **Feature-Based KD** (FitNets / Attention Transfer) — match intermediate feature maps
- **Relation-Based KD** (RKD, Park et al., 2019) — match pairwise structure in embedding space

**Core question:** How much of the teacher's accuracy (82–85%) can a 7.5× smaller student recover through distillation?

---

## Hardware Requirements

| Component | Minimum | Used here |
|-----------|---------|-----------|
| GPU VRAM  | 6 GB    | 8 GB (RTX 2080) |
| RAM       | 16 GB   | — |
| Disk      | 5 GB    | ~1.1 GB dataset + ~0.5 GB weights |

All training uses **mixed precision (AMP/FP16)**, keeping peak GPU memory at ~2–2.5 GB with batch size 32, leaving comfortable headroom on 8 GB.

---

## Project Structure

```
CV/
├── configs/
│   └── config.yaml          # all hyperparameters for every experiment
├── data/
│   └── CUB_200_2011/        # dataset (downloaded in Step 3)
├── weights/                 # pretrained ImageNet weights (downloaded in Step 5)
│   ├── resnet50_imagenet.pth
│   └── mobilenetv2_imagenet.pth
├── results/                 # auto-created; all checkpoints + CSV logs saved here
├── dataset.py               # CUB-200-2011 Dataset class, transforms, DataLoaders
├── model.py                 # ResNet-50 teacher: loads from local weights file
├── student_model.py         # MobileNetV2 student: loads from local weights file
├── distillation.py          # all 5 loss functions (HintonKD, FeatureKD, AT, RKD, Combined)
├── train_teacher.py         # Week 1: fine-tune teacher, log accuracy
├── train_student.py         # Weeks 2–5: single script for all student experiments
├── evaluate.py              # Week 5: load all checkpoints, print full results table
└── requirements.txt         # pip dependencies
```

---

## File Descriptions

### `dataset.py`
Loads CUB-200-2011 by reading the three metadata files (`images.txt`, `train_test_split.txt`, `image_class_labels.txt`). Returns PyTorch `DataLoader` objects for train and val splits. Applies standard ImageNet augmentation on train (RandomResizedCrop, HorizontalFlip, ColorJitter) and centre-crop on val.

### `model.py`
Loads ResNet-50 from a local `.pth` file (no internet required). Replaces the final fully-connected layer from 1000 classes (ImageNet) to 200 classes (CUB-200).

### `student_model.py`
Loads MobileNetV2 from a local `.pth` file. Replaces the classifier head for 200 classes. Has 3.4M parameters vs teacher's 25.6M — a 7.5× compression ratio.

### `distillation.py`
Contains all loss functions used across Weeks 3–5:

| Class | Week | What it does |
|-------|------|-------------|
| `HintonKDLoss` | 3 | KL divergence between teacher and student soft outputs at temperature T, weighted by α |
| `FeatureKDLoss` | 4 | MSE between teacher's layer3 features and student's features[14] after a 1×1 conv adapter |
| `AttentionTransferLoss` | 4 | MSE between L2-normalised spatial attention maps (sum of squared activations across channels) |
| `RKDLoss` | 5 | Huber loss on normalised pairwise distances (distance-wise) and cosine angles of triplets (angle-wise) in embedding space |
| `CombinedKDLoss` | 5 | Hinton KD + Feature matching together in one loss |

### `train_teacher.py`
Fine-tunes ResNet-50 on CUB-200 for 60 epochs. Uses AdamW + cosine LR decay + AMP. Saves the best checkpoint and logs all metrics to CSV.

### `train_student.py`
Single script for all student experiments (Weeks 2–5). Controlled by `--method` flag. Handles teacher loading and freezing, forward hook registration for feature extraction, and per-method loss selection. Saves best checkpoint and CSV log for each run.

### `evaluate.py`
Loads every saved checkpoint, runs evaluation on the val set, and prints a full comparison table (top-1, top-5, parameter count, model size, latency). Saves results to `results/final_results.csv`.

### `configs/config.yaml`
Single source of truth for all hyperparameters. Edit this file to change batch size, learning rate, epochs, temperature, alpha, beta, etc. without touching any Python code.

---

## One-Time Setup

### Step 1 — Create conda environment

```bash
conda create -n kd_project python=3.10 -y
conda activate kd_project
```

> All commands below assume this environment is active.

---

### Step 2 — Install dependencies

PyTorch must be installed separately with the CUDA index URL. Then install the rest from `requirements.txt`.

```bash
# Install PyTorch with CUDA 11.8 support
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install remaining packages
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118
```

> Replace `cu118` with your CUDA version (`cu117`, `cu121`, etc.). Check with `nvidia-smi`.

**Installed packages:**

| Package     | Version  |
|-------------|----------|
| torch       | 2.2.0    |
| torchvision | 0.17.0   |
| numpy       | 1.24.4   |
| Pillow      | 10.0.1   |
| PyYAML      | 6.0.1    |

---

### Step 3 — Download the dataset

Run from inside the `CV/` directory. No browser or login required.

```bash
mkdir -p data

# Download (~1.1 GB) — -L follows the redirect to the actual file
wget -L "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz" \
     -O data/CUB_200_2011.tgz

# Extract
tar -xzf data/CUB_200_2011.tgz -C data/
```

> No `wget`? Use `curl -L` instead:
> ```bash
> curl -L "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz" \
>      -o data/CUB_200_2011.tgz
> ```

Verify the extraction is correct:

```bash
ls data/CUB_200_2011/
# expected: images/  images.txt  train_test_split.txt  image_class_labels.txt  classes.txt
```

If you see a nested folder (`data/CUB_200_2011/CUB_200_2011/`), fix it:

```bash
mv data/CUB_200_2011/CUB_200_2011/* data/CUB_200_2011/
```

**Dataset statistics:**

| Split      | Images |
|------------|--------|
| Train      | 5,994  |
| Validation | 5,794  |
| Classes    | 200    |

---

### Step 4 — Verify GPU

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected:
```
True
NVIDIA GeForce RTX 2080
```

---

### Step 5 — Download pretrained ImageNet weights

Both models are loaded from local files to avoid permission issues with shared HuggingFace/PyTorch cache directories.

```bash
mkdir -p weights

# ResNet-50 ImageNet weights (~98 MB)
wget https://download.pytorch.org/models/resnet50-0676ba61.pth \
     -O weights/resnet50_imagenet.pth

# MobileNetV2 ImageNet weights (~14 MB)
wget https://download.pytorch.org/models/mobilenet_v2-b0353104.pth \
     -O weights/mobilenetv2_imagenet.pth
```

---

### Step 6 — Review configuration

`configs/config.yaml` controls all hyperparameters. No need to edit Python files to change settings.

```yaml
seed: 42

dataset:
  root: data/CUB_200_2011
  num_classes: 200
  img_size: 224
  batch_size: 32        # safe for 8 GB GPU with AMP; use 64 only if VRAM allows
  num_workers: 4

teacher:
  arch: resnet50
  weights_path: weights/resnet50_imagenet.pth
  epochs: 60
  lr: 1.0e-4            # small LR to preserve pretrained features
  weight_decay: 1.0e-4
  label_smoothing: 0.1  # regularisation against overconfidence
  checkpoint: results/teacher_best.pth

student:
  arch: mobilenetv2
  weights_path: weights/mobilenetv2_imagenet.pth
  epochs: 100
  lr: 1.0e-3
  weight_decay: 1.0e-4

distillation:
  temperature: 4        # controls softness of teacher distribution (sweep: 2, 4, 8)
  alpha: 0.7            # weight on KD loss vs CE loss
  beta: 0.3             # weight on feature loss vs CE loss
  lambda_d: 25.0        # RKD distance-wise loss weight
  lambda_a: 50.0        # RKD angle-wise loss weight
```

---

## Week 1 — Teacher Fine-Tuning

### What it does
Loads ResNet-50 with ImageNet pretrained weights, replaces the 1000-class head with a 200-class head, and fine-tunes on CUB-200-2011. Uses AdamW optimiser with cosine learning rate decay and label smoothing. Mixed precision (AMP) reduces memory usage and speeds up training via Tensor Cores.

### Run

```bash
python train_teacher.py
```

### What happens step by step
1. Random seeds fixed across Python, NumPy, and PyTorch for full reproducibility
2. CUB-200 train set loaded with data augmentation; val set loaded with centre-crop only
3. ResNet-50 loaded from `weights/resnet50_imagenet.pth`; final FC layer replaced for 200 classes
4. Training runs for 60 epochs — after each epoch, val top-1 and top-5 are computed
5. If val top-1 improves, checkpoint is saved to `results/teacher_best.pth`
6. All metrics appended to `results/teacher_log.csv`

### Console output
```
Device: cuda
Teacher params: 25.6M
[  1/60]  loss=3.4821  train=12.34%  val_top1=38.21%  val_top5=61.45%
[  2/60]  loss=2.9103  train=28.11%  val_top1=51.33%  val_top5=74.88%
  -> Saved best model (val_top1=51.33%)
...
[ 60/60]  loss=0.8234  train=81.20%  val_top1=83.45%  val_top5=95.12%

Done. Best val top-1: 83.45%
```

### Outputs
| File | Description |
|------|-------------|
| `results/teacher_best.pth` | Best teacher checkpoint (used by all Week 3–5 experiments) |
| `results/teacher_log.csv` | per-epoch: epoch, train_loss, train_top1, val_top1, val_top5 |

### Expected outcome
- Val top-1: **82–85%**
- Training time: ~3–4 min/epoch → ~3–4 hours total on RTX 2080

---

## Week 2 — Student Baselines

### What it does
Trains MobileNetV2 in two configurations to establish reference points before any distillation. These numbers define how much distillation actually helps.

- **Baseline-A**: random initialisation, trained from scratch with hard labels only — shows raw capacity of the small model
- **Baseline-B**: ImageNet pretrained weights, trained with hard labels only — shows the benefit of pretraining alone, independent of distillation

### Run

```bash
# Baseline A — no pretrained weights
python train_student.py --method baseline_a

# Baseline B — ImageNet pretrained
python train_student.py --method baseline_b
```

### What happens
1. Teacher is **not** loaded for baselines — only standard cross-entropy on hard one-hot labels
2. Baseline-A initialises from random weights; Baseline-B loads `weights/mobilenetv2_imagenet.pth`
3. Both train for 100 epochs with AdamW + cosine LR + AMP
4. Best checkpoint and per-epoch CSV log saved for each

### Outputs
| File | Description |
|------|-------------|
| `results/baseline_a_best.pth` | Best Baseline-A checkpoint |
| `results/baseline_a_log.csv` | per-epoch metrics for Baseline-A |
| `results/baseline_b_best.pth` | Best Baseline-B checkpoint |
| `results/baseline_b_log.csv` | per-epoch metrics for Baseline-B |

### Expected outcome
| Method | Val Top-1 |
|--------|-----------|
| Baseline-A (scratch) | ~55–60% |
| Baseline-B (pretrained) | ~75–78% |

The gap between A and B (~18%) shows how much ImageNet pretraining contributes. The gap between B and the teacher (~7%) is the target distillation must close.

---

## Week 3 — Hinton Knowledge Distillation

### What it does
Instead of training on hard one-hot labels only, the student also learns to match the teacher's full softmax output distribution. At temperature T=4, the teacher's soft outputs reveal inter-class similarities (e.g. "American Robin" and "Hermit Thrush" receive similar probabilities) — this is **dark knowledge** that hard labels discard.

**Loss:** `L = α · T² · KL(q_teacher || p_student) + (1-α) · CE`

The T² factor compensates for gradient shrinkage at high temperature.

### Run

```bash
# Default: T=4, alpha=0.7
python train_student.py --method hinton

# Temperature sweep
python train_student.py --method hinton --temperature 2
python train_student.py --method hinton --temperature 8

# Alpha sweep at T=4
python train_student.py --method hinton --alpha 0.5
python train_student.py --method hinton --alpha 0.9
```

### What happens
1. Frozen teacher loaded from `results/teacher_best.pth`
2. For each batch: teacher runs in `torch.no_grad()` to produce soft logits; student logits computed normally
3. `HintonKDLoss` computes KL divergence at temperature T plus cross-entropy at T=1, combined by α
4. Temperature and alpha can be overridden from CLI without editing config

### Outputs
| File | Description |
|------|-------------|
| `results/hinton_best.pth` | Best checkpoint at default T=4 |
| `results/hinton_T2_best.pth` | Best checkpoint at T=2 |
| `results/hinton_T8_best.pth` | Best checkpoint at T=8 |
| `results/hinton_log.csv` | per-epoch metrics |

### Expected outcome
| Temperature | Val Top-1 |
|-------------|-----------|
| T=2 | ~77–79% |
| T=4 | ~78–81% |
| T=8 | ~77–80% |

T=4 is typically optimal for fine-grained tasks. Higher T softens distributions too much; lower T approaches hard labels.

---

## Week 4 — Feature-Based Distillation

### What it does
Rather than only matching the teacher's final output, the student also matches the teacher's **intermediate feature maps**. The teacher's middle layers capture part-level representations (beaks, wings, eyes for birds) that the output layer alone does not convey.

**Feature KD:** A learnable 1×1 conv adapter projects student's 96-channel features to match teacher's 1024-channel features at the same 14×14 spatial resolution. Loss is MSE between projected and teacher features.

**Attention Transfer:** Instead of matching raw feature values, match the spatial attention map — where in the image each network is looking. Attention map = sum of squared activations across channels, L2-normalised. The student learns to focus on the same image regions as the teacher even if exact feature values differ.

**Hook locations:**
- Teacher: `layer3` → shape `[B, 1024, 14, 14]`
- Student: `features[14]` → shape `[B, 96, 14, 14]`
- Both at 14×14 spatial resolution — no spatial adapter needed

### Run

```bash
# FitNets-style L2 feature matching
python train_student.py --method feature

# Attention Transfer
python train_student.py --method attention
```

### What happens
1. Forward hooks registered on both teacher and student before training begins
2. Each forward pass automatically populates a `cache` dict with feature tensors
3. `FeatureKDLoss` applies the 1×1 conv adapter to student features and computes MSE vs teacher features
4. `AttentionTransferLoss` computes L2-normalised attention maps for both and minimises their MSE
5. Adapter parameters are included in the optimiser alongside student parameters

### Outputs
| File | Description |
|------|-------------|
| `results/feature_best.pth` | Best Feature KD checkpoint |
| `results/feature_log.csv` | per-epoch metrics |
| `results/attention_best.pth` | Best Attention Transfer checkpoint |
| `results/attention_log.csv` | per-epoch metrics |

### Expected outcome
| Method | Val Top-1 |
|--------|-----------|
| Feature KD | ~79–82% |
| Attention Transfer | ~79–81% |

Feature KD often outperforms pure response KD because it guides the student's internal representations, not just its final predictions.

---

## Week 5 — Relation-Based KD, Combined Loss, Final Evaluation

### What it does

**RKD:** Instead of matching individual activations point-by-point, match the *geometric relationships* between samples in embedding space. If the teacher places "American Robin" and "Hermit Thrush" close together and "Pelican" far away, the student should preserve this same structure — even if absolute embedding values differ.

Two terms:
- **Distance-wise:** Huber loss on normalised pairwise distances between all sample pairs in the batch
- **Angle-wise:** Huber loss on cosine angles for all sample triplets

Huber loss is used instead of MSE for robustness to outlier distances.

**Combined KD:** Runs Hinton response-based KD and feature matching simultaneously in a single loss:
`L = α · T² · KL + β · MSE(adapter(F_s), F_t) + (1-α-β) · CE`

### Run

```bash
# Relational KD
python train_student.py --method rkd

# Combined: response KD + feature matching
python train_student.py --method combined
```

### Final evaluation — all methods

Once all experiments are complete, run:

```bash
python evaluate.py
```

This automatically finds every checkpoint in `results/`, evaluates each on the val set, and prints a complete comparison table.

### What `evaluate.py` measures per model
- **Top-1 accuracy** — primary metric
- **Top-5 accuracy** — top-5 prediction contains correct class
- **Parameter count (M)** — model size in millions of parameters
- **Model size (MB)** — disk footprint at float32
- **Inference latency (ms)** — averaged over 200 forward passes, single image, on GPU

### Console output from `evaluate.py`
```
Method                    Top-1  Top-5  Params(M)  Size(MB)  Lat(ms)
------------------------------------------------------------------
  Teacher (ResNet-50)     83.45  96.12      25.56      98.0    18.20
  Baseline-A (scratch)    57.21  79.33       3.41      13.1     4.10
  Baseline-B (pretrain)   76.88  92.44       3.41      13.1     4.10
  Hinton KD (T=2)         78.33  93.21       3.41      13.1     4.11
  Hinton KD (T=4)         79.92  94.10       3.41      13.1     4.12
  Hinton KD (T=8)         78.87  93.88       3.41      13.1     4.12
  Feature KD              81.33  94.87       3.41      13.1     4.13
  Attention Transfer      80.11  94.45       3.41      13.1     4.11
  RKD                     79.55  94.02       3.41      13.1     4.10
  Combined KD             82.01  95.44       3.41      13.1     4.14

Saved: results/final_results.csv
```

### Outputs
| File | Description |
|------|-------------|
| `results/rkd_best.pth` | Best RKD checkpoint |
| `results/rkd_log.csv` | per-epoch metrics |
| `results/combined_best.pth` | Best Combined KD checkpoint |
| `results/combined_log.csv` | per-epoch metrics |
| `results/final_results.csv` | Full comparison table, all methods |

### Expected outcome
| Method | Val Top-1 | vs Teacher |
|--------|-----------|-----------|
| Teacher (ResNet-50) | ~83–85% | — |
| Baseline-A | ~55–60% | −25% |
| Baseline-B | ~75–78% | −8% |
| Hinton KD (T=4) | ~78–81% | −4% |
| Feature KD | ~79–82% | −3% |
| Attention Transfer | ~79–81% | −3% |
| RKD | ~78–81% | −4% |
| Combined KD | ~80–83% | −2% |

Combined KD typically recovers the most accuracy while keeping the student at 3.4M parameters vs teacher's 25.6M.

---

## Full Run Order (All Weeks)

```bash
# ── One-time setup ────────────────────────────────────────────────
conda create -n kd_project python=3.10 -y && conda activate kd_project
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118

mkdir -p data weights
wget -L "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz" \
     -O data/CUB_200_2011.tgz && tar -xzf data/CUB_200_2011.tgz -C data/
wget https://download.pytorch.org/models/resnet50-0676ba61.pth \
     -O weights/resnet50_imagenet.pth
wget https://download.pytorch.org/models/mobilenet_v2-b0353104.pth \
     -O weights/mobilenetv2_imagenet.pth

# ── Week 1 ────────────────────────────────────────────────────────
python train_teacher.py

# ── Week 2 ────────────────────────────────────────────────────────
python train_student.py --method baseline_a
python train_student.py --method baseline_b

# ── Week 3 ────────────────────────────────────────────────────────
python train_student.py --method hinton
python train_student.py --method hinton --temperature 2
python train_student.py --method hinton --temperature 8
python train_student.py --method hinton --alpha 0.5
python train_student.py --method hinton --alpha 0.9

# ── Week 4 ────────────────────────────────────────────────────────
python train_student.py --method feature
python train_student.py --method attention

# ── Week 5 ────────────────────────────────────────────────────────
python train_student.py --method rkd
python train_student.py --method combined
python evaluate.py
```

---

## Troubleshooting

**`CUDA out of memory`**
```yaml
# configs/config.yaml
batch_size: 16
```

**`FileNotFoundError: data/CUB_200_2011/images.txt`**

Dataset extracted into wrong path. Check for nested folder:
```bash
ls data/CUB_200_2011/
# if you see another CUB_200_2011/ folder inside:
mv data/CUB_200_2011/CUB_200_2011/* data/CUB_200_2011/
```

**`KeyError: teacher checkpoint not found` when running train_student.py**

Week 1 must complete first. The file `results/teacher_best.pth` must exist before running any distillation experiment.

**DataLoader hanging / `num_workers` warning**
```yaml
# configs/config.yaml
num_workers: 0
```

**GPU not being used / slow training**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# must print True
nvidia-smi   # check GPU utilisation in a second terminal during training
```

---

## Memory Budget (RTX 2080, 8 GB)

| Component | Teacher (FP32) | Student + AMP |
|-----------|---------------|---------------|
| Model weights | ~100 MB | ~14 MB |
| Gradients | ~100 MB | ~14 MB |
| AdamW states | ~200 MB | ~28 MB |
| Activations (batch 32) | ~1,760 MB | ~440 MB |
| **Real usage** | **~3.5 GB** | **~0.8 GB** |

Teacher training peaks at ~3.5 GB (well within 8 GB). Student training with distillation (teacher loaded frozen + student) peaks at ~2.5 GB.

---

## Data Preprocessing

| Split | Operations |
|-------|-----------|
| Train | RandomResizedCrop(224) → RandomHorizontalFlip → ColorJitter(b=0.4, c=0.4, s=0.4) → Normalize |
| Val   | Resize(256) → CenterCrop(224) → Normalize |

Normalisation: ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`. Both teacher and student use identical preprocessing so comparisons are fair.

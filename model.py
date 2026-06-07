import torch
import torch.nn as nn
from torchvision.models import resnet50


def load_teacher(num_classes=200, weights_path="weights/resnet50_imagenet.pth"):
    model = resnet50()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    # If checkpoint already has the fine-tuned head, replace FC before loading
    ckpt_classes = state["fc.weight"].shape[0]
    if ckpt_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, ckpt_classes)
    model.load_state_dict(state)
    # Adjust head if target num_classes differs from checkpoint
    if model.fc.out_features != num_classes:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

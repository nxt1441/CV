import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


def load_student(num_classes=200, weights_path=None):
    model = mobilenet_v2()
    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

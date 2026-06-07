from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_transforms(img_size, train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class CUB200(Dataset):
    def __init__(self, root, train=True, transform=None):
        root = Path(root)
        self.transform = transform

        id_to_path  = {int(l.split()[0]): l.split()[1]
                       for l in open(root / "images.txt")}
        id_to_split = {int(l.split()[0]): int(l.split()[1])
                       for l in open(root / "train_test_split.txt")}
        id_to_label = {int(l.split()[0]): int(l.split()[1]) - 1
                       for l in open(root / "image_class_labels.txt")}

        self.samples = [
            (root / "images" / id_to_path[i], id_to_label[i])
            for i, split in id_to_split.items()
            if (split == 1) == train
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img) if self.transform else img, label


def get_loaders(root, img_size=224, batch_size=64, num_workers=4):
    train_ds = CUB200(root, train=True,  transform=get_transforms(img_size, True))
    val_ds   = CUB200(root, train=False, transform=get_transforms(img_size, False))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl

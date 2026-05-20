import copy
import os
import time
from dataclasses import dataclass
from typing import Callable

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

torch.manual_seed(42)

CONFIG = {
    "image_dir": "3.output",
    "mask_dir": "5.compiled_masks",
    "batch_size": 8,
    "epochs": 500,
    "patience": 15,
    "learning_rate": 1e-4,
    "num_classes": 3,
    "val_split": 0.2,
    "weights_save_path": "best_unet_model.pth",
}


@dataclass
class TrainCallbacks:
    on_epoch_end:  Callable[[int, float, float], None] | None = None
    on_no_improve: Callable[[int], None]               | None = None
    on_early_stop: Callable[[int], None]               | None = None
    on_log:        Callable[[str], None]               | None = None
    should_stop:   Callable[[], bool]                  | None = None


class SolderDefectDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.image_dir, self.images[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.mask_dir, self.masks[idx]), cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = image.float() / 255.0
        mask = mask.long()
        return image, mask


def get_train_transforms():
    return A.Compose([
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.RandomCrop(width=512, height=512, p=1.0),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2()
    ])


def get_val_transforms():
    return A.Compose([
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.CenterCrop(width=512, height=512, p=1.0),
        ToTensorV2()
    ])


def _log(callbacks: TrainCallbacks | None, msg: str):
    if callbacks and callbacks.on_log:
        callbacks.on_log(msg)
    else:
        print(msg)


def train_model(config: dict | None = None, callbacks: TrainCallbacks | None = None):
    if config is None:
        config = CONFIG

    # Suppress tqdm per-batch bars when the UI is driving (callbacks provided)
    tqdm_disabled = callbacks is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(callbacks, f"[SYSTEM] Training on: {device}")

    full_dataset = SolderDefectDataset(config["image_dir"], config["mask_dir"])

    val_size = int(len(full_dataset) * config["val_split"])
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    val_dataset.dataset = copy.copy(full_dataset)
    train_dataset.dataset.transform = get_train_transforms()
    val_dataset.dataset.transform = get_val_transforms()

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                            num_workers=4, pin_memory=True)

    _log(callbacks, f"[DATA] Train: {train_size} | Val: {val_size}")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=config["num_classes"]
    ).to(device)

    class_weights = torch.tensor([0.1, 1.0, 5.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scaler = GradScaler()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, config["epochs"] + 1):
        # Check for stop request from UI
        if callbacks and callbacks.should_stop and callbacks.should_stop():
            _log(callbacks, f"[SYSTEM] Training stopped by user at epoch {epoch}.")
            break

        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']} [Train]",
                    leave=False, disable=tqdm_disabled)

        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            if not tqdm_disabled:
                loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                with autocast(device_type='cuda'):
                    logits = model(images)
                    loss = criterion(logits, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if callbacks and callbacks.on_epoch_end:
            callbacks.on_epoch_end(epoch, avg_train_loss, avg_val_loss)
        else:
            print(f"Epoch {epoch:03d}/{config['epochs']} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            tmp_path = config["weights_save_path"] + ".tmp"
            torch.save(model.state_dict(), tmp_path)
            for _attempt in range(5):
                try:
                    os.replace(tmp_path, config["weights_save_path"])
                    break
                except PermissionError:
                    if _attempt == 4:
                        raise
                    time.sleep(1)

            _log(callbacks, f"[SAVE] Model saved → {config['weights_save_path']}")
        else:
            epochs_no_improve += 1
            if callbacks and callbacks.on_no_improve:
                callbacks.on_no_improve(epochs_no_improve)
            else:
                print(f" -> [WARNING] No improvement for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= config["patience"]:
                if callbacks and callbacks.on_early_stop:
                    callbacks.on_early_stop(epoch)
                else:
                    print(f"\n[SYSTEM] Early Stopping at epoch {epoch}.")
                break

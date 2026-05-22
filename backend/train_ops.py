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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from backend.prepare_ops import preprocess_image

torch.manual_seed(42)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

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
    "auto_preprocess": False,
    "include_original_variant": True,
    "preprocess_profiles": [],
}


@dataclass
class TrainCallbacks:
    on_epoch_end:  Callable[[int, float, float], None] | None = None
    on_no_improve: Callable[[int], None]               | None = None
    on_early_stop: Callable[[int], None]               | None = None
    on_log:        Callable[[str], None]               | None = None
    should_stop:   Callable[[], bool]                  | None = None


def _enabled_profiles(profiles: list[dict] | None) -> list[dict]:
    enabled = []
    for idx, profile in enumerate(profiles or [], start=1):
        if not profile.get("enabled", False):
            continue
        enabled.append(
            {
                "name": profile.get("name") or f"Preset {idx}",
                "blur_kernel": int(profile.get("blur_kernel", 3)),
                "clahe_clip": float(profile.get("clahe_clip", 12.0)),
                "clahe_tile_w": int(profile.get("clahe_tile_w", 32)),
                "clahe_tile_h": int(profile.get("clahe_tile_h", 32)),
            }
        )
    return enabled


def _collect_sample_pairs(image_dir: str, mask_dir: str) -> list[tuple[str, str]]:
    image_by_stem = {}
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            stem = os.path.splitext(filename)[0]
            image_by_stem[stem] = os.path.join(image_dir, filename)

    sample_pairs = []
    missing_images = []
    for mask_name in sorted(os.listdir(mask_dir)):
        if not mask_name.lower().endswith(".png"):
            continue
        stem = os.path.splitext(mask_name)[0]
        image_path = image_by_stem.get(stem)
        if image_path is None:
            missing_images.append(mask_name)
            continue
        sample_pairs.append((image_path, os.path.join(mask_dir, mask_name)))

    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(
            f"Could not find matching images for {len(missing_images)} mask(s): {preview}"
        )

    if not sample_pairs:
        raise FileNotFoundError("No matching image/mask pairs were found for training.")

    return sample_pairs


class SolderDefectDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, str]],
        transform=None,
        preprocess_profiles: list[dict] | None = None,
        include_original_variant: bool = True,
        auto_preprocess: bool = False,
    ):
        self.samples = samples
        self.transform = transform
        self.auto_preprocess = auto_preprocess
        self.variant_profiles = _enabled_profiles(preprocess_profiles) if auto_preprocess else []
        self.variant_specs = []

        if include_original_variant or not self.variant_profiles:
            self.variant_specs.append(None)
        self.variant_specs.extend(self.variant_profiles)

    def __len__(self):
        return len(self.samples) * len(self.variant_specs)

    def __getitem__(self, idx):
        sample_idx = idx // len(self.variant_specs)
        variant_idx = idx % len(self.variant_specs)
        image_path, mask_path = self.samples[sample_idx]
        variant = self.variant_specs[variant_idx]

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        if self.auto_preprocess and variant is not None:
            image = preprocess_image(
                image,
                blur_kernel=variant["blur_kernel"],
                clahe_clip=variant["clahe_clip"],
                clahe_tile_w=variant["clahe_tile_w"],
                clahe_tile_h=variant["clahe_tile_h"],
            )

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image/mask size mismatch for '{os.path.basename(image_path)}': "
                f"{image.shape[:2]} vs {mask.shape[:2]}"
            )

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert to tensors and normalize.
        # If `ToTensorV2()` is part of the pipeline, albumentations already
        # returns torch.Tensors scaled to [0,1]. Avoid dividing by 255 again
        # in that case. If we have numpy arrays, convert and scale here.
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        else:
            image = image.float()

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
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

    base_samples = _collect_sample_pairs(config["image_dir"], config["mask_dir"])
    val_size = int(len(base_samples) * config["val_split"])
    if len(base_samples) > 1:
        val_size = max(1, min(val_size, len(base_samples) - 1))
    if val_size == 0:
        raise ValueError("Training needs at least 2 matched image/mask pairs for a train/validation split.")
    train_size = len(base_samples) - val_size

    generator = torch.Generator().manual_seed(42)
    permutation = torch.randperm(len(base_samples), generator=generator).tolist()
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:]
    train_samples = [base_samples[i] for i in train_indices]
    val_samples = [base_samples[i] for i in val_indices]

    train_dataset = SolderDefectDataset(
        train_samples,
        transform=get_train_transforms(),
        preprocess_profiles=config.get("preprocess_profiles"),
        include_original_variant=config.get("include_original_variant", True),
        auto_preprocess=config.get("auto_preprocess", False),
    )
    val_dataset = SolderDefectDataset(
        val_samples,
        transform=get_val_transforms(),
        preprocess_profiles=None,
        include_original_variant=True,
        auto_preprocess=config.get("auto_preprocess", False),
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                            num_workers=4, pin_memory=True)

    enabled_profiles = _enabled_profiles(config.get("preprocess_profiles"))
    variant_count = len(train_dataset.variant_specs)
    _log(callbacks, f"[DATA] Base samples -> Train: {train_size} | Val: {val_size}")
    if config.get("auto_preprocess", False):
        _log(callbacks, f"[DATA] Auto preprocessing enabled | Train variants per sample: {variant_count}")
        if enabled_profiles:
            names = ", ".join(profile["name"] for profile in enabled_profiles)
            _log(callbacks, f"[DATA] Active presets: {names}")
    _log(callbacks, f"[DATA] Expanded train samples: {len(train_dataset)}")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=config["num_classes"]
    ).to(device)

    class_weights = torch.tensor([0.1, 1.0, 5.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    amp_enabled = device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

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

            with autocast(device_type=device.type, enabled=amp_enabled):
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
                with autocast(device_type=device.type, enabled=amp_enabled):
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

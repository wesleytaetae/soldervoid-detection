import cv2
import numpy as np


CLASS_NAMES = {
    0: "Background",
    1: "Solder",
    2: "Void",
}


def load_mask(mask_path: str) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    return mask


def compute_iou_metrics(pred_mask: np.ndarray, true_mask: np.ndarray, num_classes: int = 3) -> dict:
    if pred_mask.shape != true_mask.shape:
        raise ValueError(
            f"Prediction and ground-truth masks must have the same shape, got "
            f"{pred_mask.shape} vs {true_mask.shape}."
        )

    per_class = []
    valid_ious = []

    for class_id in range(num_classes):
        pred_class = pred_mask == class_id
        true_class = true_mask == class_id

        intersection = int(np.logical_and(pred_class, true_class).sum())
        union = int(np.logical_or(pred_class, true_class).sum())
        iou = None if union == 0 else intersection / union
        if iou is not None:
            valid_ious.append(iou)

        per_class.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, f"Class {class_id}"),
                "intersection": intersection,
                "union": union,
                "iou": iou,
            }
        )

    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else None
    return {
        "per_class": per_class,
        "mean_iou": mean_iou,
    }

import cv2
import json
import os
import numpy as np


def compile_dataset_masks(json_dir: str, mask_output_dir: str, on_file_done=None) -> None:
    """Rasterises LabelMe JSON polygons into multi-class uint8 PNG masks."""
    os.makedirs(mask_output_dir, exist_ok=True)

    for filename in os.listdir(json_dir):
        if not filename.endswith('.json'):
            continue

        with open(os.path.join(json_dir, filename), 'r') as f:
            data = json.load(f)

        img_height = data.get('imageHeight', 1024)
        img_width = data.get('imageWidth', 1024)
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        solder_polygons, void_polygons = [], []
        for shape in data['shapes']:
            label = shape['label'].strip().lower()
            points = np.array(shape['points'], dtype=np.int32)
            if label == 'solder':
                solder_polygons.append(points)
            elif label in ('solder void', 'void'):
                void_polygons.append(points)
            else:
                print(f"[WARNING] Unrecognized label '{label}' in {filename}. Skipping.")

        for poly in solder_polygons:
            cv2.fillPoly(mask, [poly], color=1)
        for poly in void_polygons:
            cv2.fillPoly(mask, [poly], color=2)

        mask_filename = filename.replace('.json', '.png')
        cv2.imwrite(os.path.join(mask_output_dir, mask_filename), mask)

        if on_file_done:
            on_file_done(filename)

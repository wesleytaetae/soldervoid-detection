import os
from pathlib import Path
from PIL import Image


def resize_image(image: Image.Image, target_size: tuple[int, int] = (1024, 1024)) -> Image.Image:
    """Center-crop a PIL image to a square and resize it to the training geometry."""
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    upper = (h - side) // 2
    cropped_img = image.crop((left, upper, left + side, upper + side))
    return cropped_img.resize(target_size, Image.Resampling.LANCZOS)


def process_image_directory(
    input_dir_path: str,
    output_dir_path: str,
    target_size: tuple[int, int] = (1024, 1024),
    on_file_done=None,
):
    """Auto center-crops each image to a square then resizes to target_size."""
    input_dir = Path(input_dir_path)
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}

    for file_path in input_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            try:
                with Image.open(file_path) as img:
                    resized_img = resize_image(img, target_size=target_size)
                    resized_img.save(output_dir / file_path.name)
                if on_file_done:
                    on_file_done(file_path.name)
            except Exception as e:
                print(f"Failed to process {file_path.name}: {e}")

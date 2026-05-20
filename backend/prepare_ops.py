import cv2
import os


def prepare_images_for_labelme(
    input_dir: str,
    output_dir: str,
    blur_kernel: int = 3,
    clahe_clip: float = 12.0,
    clahe_tile_w: int = 32,
    clahe_tile_h: int = 32,
    on_file_done=None,
):
    """Applies median blur + CLAHE to maximise human visibility in LabelMe."""
    os.makedirs(output_dir, exist_ok=True)

    if blur_kernel % 2 == 0:
        blur_kernel += 1

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile_w, clahe_tile_h))

    for filename in os.listdir(input_dir):
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            continue

        img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        denoised = cv2.medianBlur(img, blur_kernel)
        enhanced = clahe.apply(denoised)

        cv2.imwrite(os.path.join(output_dir, filename), enhanced)
        if on_file_done:
            on_file_done(filename)


def preview_single_image(
    image_path: str,
    blur_kernel: int = 3,
    clahe_clip: float = 12.0,
    clahe_tile_w: int = 32,
    clahe_tile_h: int = 32,
) -> tuple[bytes, bytes]:
    """Returns (original_png_bytes, enhanced_png_bytes) for a single image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    _, orig_buf = cv2.imencode(".png", img)

    if blur_kernel % 2 == 0:
        blur_kernel += 1
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile_w, clahe_tile_h))
    enhanced = clahe.apply(cv2.medianBlur(img, blur_kernel))
    _, enh_buf = cv2.imencode(".png", enhanced)

    return bytes(orig_buf), bytes(enh_buf)

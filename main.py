import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <command>")
        print("Commands: resize | prepare | mask | train | all")
        return

    match sys.argv[1]:
        case "resize":
            from backend.resize_ops import process_image_directory
            process_image_directory("1.input", "2.resize")
        case "prepare":
            from backend.prepare_ops import prepare_images_for_labelme
            prepare_images_for_labelme("2.resize", "3.output")
        case "mask":
            from backend.mask_ops import compile_dataset_masks
            compile_dataset_masks("4.json_labels", "5.compiled_masks")
        case "train":
            from backend.train_ops import train_model, CONFIG
            train_model(CONFIG)
        case "all":
            from backend.resize_ops import process_image_directory
            from backend.prepare_ops import prepare_images_for_labelme
            process_image_directory("1.input", "2.resize")
            prepare_images_for_labelme("2.resize", "3.output")
        case _:
            print(f"Unknown command: {sys.argv[1]}")
            print("Commands: resize | prepare | mask | train | all")


if __name__ == "__main__":
    main()

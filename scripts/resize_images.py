import os
import argparse
from pathlib import Path
from PIL import Image


def resize_image(image_path: str, output_dir: str, resolution: int):
    """
    Resize an image to the given resolution and save it.

    :param image_path: Path to the input image.
    :param output_dir: Directory to save the resized image.
    :param resolution: Target resolution (width and height will be the same).
    """
    os.makedirs(output_dir, exist_ok=True)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((resolution, resolution), Image.BICUBIC)
        img.save(os.path.join(output_dir, Path(image_path).name))


def process_directory(input_dir: str, output_dir: str, resolution: int):
    """
    Process all images in a directory, resizing them to the specified resolution.

    :param input_dir: Directory containing the input images.
    :param output_dir: Directory to save the resized images.
    :param resolution: Target resolution (width and height will be the same).
    """
    for image_file in Path(input_dir).glob("*.jpg"):
        resize_image(str(image_file), output_dir, resolution)


def main():
    parser = argparse.ArgumentParser(description="Resize images in a directory")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory for images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for resized images")
    parser.add_argument("--resolution", type=int, default=512, help="Target resolution")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.resolution)


if __name__ == "__main__":
    main()

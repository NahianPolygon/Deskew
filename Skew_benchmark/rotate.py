import os
import math
import cv2
import numpy as np
from typing import Tuple, Union
import random
import csv

def rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def add_random_rotation_to_folder(input_dir: str, output_dir: str, min_angle: float = -15.0, max_angle: float = 15.0, csv_writer=None):
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    processed_count = 0

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(valid_exts):
            continue

        input_path = os.path.join(input_dir, filename)
        image = cv2.imread(input_path)

        if image is None:
            print(f"Skipping unreadable image: {filename}")
            continue

        random_angle = random.uniform(min_angle, max_angle)

        rotated = rotate(image, random_angle, (255, 255, 255))

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, rotated)

        if csv_writer:
            csv_writer.writerow({
                'filename': filename,
                'applied_angle': round(random_angle, 2),
                'saved_shape': str(rotated.shape)
            })

        print(f"{filename}: applied {random_angle:.2f}° rotation | saved shape={rotated.shape}")
        processed_count += 1

    print(f"Processed {processed_count} images in {input_dir}")

def main():
    directories = [
        ("front_images", "front_angled", "front_rotations.csv"),
        ("back_images", "back_angled", "back_rotations.csv")
    ]

    total_processed = 0

    for input_dir, output_dir, csv_file in directories:
        if not os.path.exists(input_dir):
            print(f"Warning: Input directory '{input_dir}' does not exist. Skipping...")
            continue

        print(f"\nProcessing {input_dir} -> {output_dir}")
        print("=" * 50)

        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'applied_angle', 'saved_shape']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            add_random_rotation_to_folder(input_dir, output_dir, min_angle=-15.0, max_angle=15.0, csv_writer=writer)

        if os.path.exists(output_dir):
            processed = len([f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
            total_processed += processed
            print(f"Created {processed} rotated images in {output_dir}")
            print(f"Saved rotation data to {csv_file}")

    print(f"\n{'='*60}")
    print(f"SUCCESS: Total {total_processed} images processed with random rotations!")
    print(f"Output directories: front_angled/, back_angled/")
    print(f"CSV files: front_rotations.csv, back_rotations.csv")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

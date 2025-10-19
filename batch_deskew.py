import os
import math
import cv2
import numpy as np
from deskew import determine_skew
from typing import Tuple, Union

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

def simple_orientation_check(image: np.ndarray) -> bool:
    """Check if image appears upside down. Returns True if should rotate 180°."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Apply threshold to get text regions
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binary.shape

    # Compare content in top 1/3 vs bottom 1/3
    top_third = binary[:h//3, :]
    bottom_third = binary[2*h//3:, :]

    top_content = np.sum(top_third == 255) / (top_third.shape[0] * top_third.shape[1])
    bottom_content = np.sum(bottom_third == 255) / (bottom_third.shape[0] * bottom_third.shape[1])

    # If bottom has significantly more content than top, might be upside down
    return bottom_content > top_content * 1.8

def deskew_folder(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(valid_exts):
            continue

        input_path = os.path.join(input_dir, filename)
        image = cv2.imread(input_path)

        if image is None:
            print(f"Skipping unreadable image: {filename}")
            continue

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        angle = determine_skew(grayscale)

        # Deskew the image
        rotated = rotate(image, angle, (0, 0, 0))
        h, w = rotated.shape[:2]

        # If image is taller than wide, rotate to make it landscape
        if h > w:
            if angle > 0:
                rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)
                rotation_applied = "90° CW"
            elif angle < 0:
                rotated = cv2.rotate(rotated, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotation_applied = "90° CCW"
            else:
                # For zero angle, try both orientations and check which looks better
                option_cw = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)
                option_ccw = cv2.rotate(rotated, cv2.ROTATE_90_COUNTERCLOCKWISE)

                cw_upside_down = simple_orientation_check(option_cw)
                ccw_upside_down = simple_orientation_check(option_ccw)

                if cw_upside_down and not ccw_upside_down:
                    rotated = option_ccw
                    rotation_applied = "90° CCW (CW appeared upside down)"
                elif ccw_upside_down and not cw_upside_down:
                    rotated = option_cw
                    rotation_applied = "90° CW (CCW appeared upside down)"
                else:
                    rotated = option_cw
                    rotation_applied = "90° CW (default for zero angle)"

            print(f"  -> Applied {rotation_applied} for angle {angle:.2f}°")

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, rotated)

        print(f"{filename}: skew={angle:.2f}° | saved shape={rotated.shape}")

# Usage
input_folder = "images_new"
output_folder = "deskewed_images"
deskew_folder(input_folder, output_folder)
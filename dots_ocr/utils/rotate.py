from PIL import Image
import cv2
import numpy as np
import pytesseract
import re
import os
from deskew import determine_skew
from loguru import logger

def detect_orientation_tesseract(image):
    osd = pytesseract.image_to_osd(image)
    angle = int(re.search(r'Rotate: (\d+)', osd).group(1))
    return angle


def rotate_image_by_angle(image, angle):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def deskew_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(gray)

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated, angle


def auto_rotate_and_deskew(pil_image: Image.Image) -> Image.Image:
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    try:
        orientation_angle = detect_orientation_tesseract(image)
    except Exception as e:
        orientation_angle = 0
        logger.warning(f"Failed to detect orientation using Tesseract: {e}")

    image = rotate_image_by_angle(image, orientation_angle)
    image, skew_angle = deskew_image(image)
    if orientation_angle != 0 or (skew_angle is not None and abs(skew_angle) >= 5):
        logger.info(f"Detected orientation angle: {orientation_angle} degrees. Detected skew angle: {skew_angle:.2f} degrees.")
    result_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return result_pil

if __name__ == "__main__":
    input_path = "test_rotate.png"
    output_path = "test_rotate_output.png"

    pil_image = Image.open(input_path)

    output_pil_image = auto_rotate_and_deskew(pil_image)
    output_pil_image.save(output_path)
    print(f"✅ saved output image to {output_path}")
    print("-" * 40)
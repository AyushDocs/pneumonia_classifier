import base64
import io

import cv2
import numpy as np
from PIL import Image


def is_valid_xray(image):
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    std_dev = np.std(img_gray)
    if std_dev < 30:
        return False, "Low contrast detected. Likely not an X-ray."
    corner_avg = (int(img_gray[0, 0]) + int(img_gray[0, -1]) + int(img_gray[-1, 0]) + int(img_gray[-1, -1])) / 4
    if corner_avg > 150:
        return False, "Input image detected as non-medical or noisy."
    return True, "Valid X-ray."

def auto_crop_xray(image: Image.Image) -> Image.Image:
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        h_m, w_m = img_gray.shape
        x, y = max(0, x-10), max(0, y-10)
        w, h = min(w_m-x, w+20), min(h_m-y, h+20)
        return image.crop((x, y, x+w, y+h))
    return image

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

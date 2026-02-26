import numpy as np
import cv2


def augment_image(img: np.ndarray) -> np.ndarray:
    """
    對已預處理的圖片（float32, [0,1]）進行隨機增強。
    增強：水平翻轉、亮度/對比調整、輕微旋轉、隨機裁剪再 resize
    """
    result = img.copy()

    if np.random.rand() > 0.5:
        result = np.fliplr(result)

    brightness = np.random.uniform(0.8, 1.2)
    result = np.clip(result * brightness, 0.0, 1.0)

    contrast = np.random.uniform(0.8, 1.2)
    result = np.clip((result - 0.5) * contrast + 0.5, 0.0, 1.0)

    angle = np.random.uniform(-15, 15)
    h, w = result.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    result = cv2.warpAffine(result, M, (w, h))

    scale = np.random.uniform(0.8, 1.0)
    crop_h, crop_w = int(h * scale), int(w * scale)
    top = np.random.randint(0, h - crop_h + 1)
    left = np.random.randint(0, w - crop_w + 1)
    cropped = result[top:top+crop_h, left:left+crop_w]
    result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)

    return result.astype(np.float32)

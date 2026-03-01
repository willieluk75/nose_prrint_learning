# src/data/preprocessor.py
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener

# Register HEIF opener at module level
register_heif_opener()

TARGET_SIZE = (224, 224)


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    將 numpy array 圖片 resize 至 224x224 並正規化至 [0, 1]。

    Args:
        img: shape (H, W, 3), dtype uint8
    Returns:
        shape (224, 224, 3), dtype float32
    """
    resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def load_and_preprocess(image_path: str) -> np.ndarray:
    """
    從檔案路徑讀取圖片並預處理。

    Raises:
        FileNotFoundError: 檔案不存在
        ValueError: 無法讀取圖片
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Step 1: Try PIL first (supports HEIC/HEIF)
    try:
        img_pil = Image.open(str(path))
        # Convert to RGB NumPy array
        img_rgb = np.array(img_pil.convert('RGB'))
        return preprocess_image(img_rgb)
    except Exception:
        # PIL failed, fall through to OpenCV
        pass

    # Step 2: OpenCV fallback for JPEG/PNG
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocess_image(img_rgb)

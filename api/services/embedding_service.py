import numpy as np
import cv2
from src.data.preprocessor import preprocess_image
from src.inference.embedder import Embedder


class EmbeddingService:
    """
    封裝圖片 → embedding 的 ML pipeline。
    供 FastAPI 路由使用。
    """

    def __init__(self, weights_path: str = None, embedding_dim: int = 256):
        self.embedder = Embedder(weights_path=weights_path, embedding_dim=embedding_dim)

    def image_bytes_to_embedding(self, image_bytes: bytes) -> np.ndarray:
        """
        將圖片 bytes（從 HTTP request 接收）轉換為 embedding 向量。

        Args:
            image_bytes: 圖片的原始 bytes
        Returns:
            shape (256,), L2 normalized float32
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Cannot decode image bytes")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_image(img_rgb)
        return self.embedder.embed(preprocessed)

import numpy as np
import tensorflow as tf
from src.models.backbone import build_backbone
from src.models.embedding_head import build_embedding_head


class Embedder:
    """從圖片生成 L2-normalized embedding 向量"""

    def __init__(self, weights_path: str = None, embedding_dim: int = 256):
        self.backbone = build_backbone(trainable=False)
        self.head = build_embedding_head(input_dim=1280, embedding_dim=embedding_dim)
        if weights_path:
            self.head.load_weights(weights_path)

    def embed(self, img: np.ndarray) -> np.ndarray:
        """
        將單張預處理圖片（224x224x3 float32）轉換為 embedding。
        Returns: shape (embedding_dim,), L2 normalized
        """
        img_batch = np.expand_dims(img, axis=0)
        features = self.backbone(img_batch, training=False)
        embedding = self.head(features, training=False)
        return embedding.numpy()[0]

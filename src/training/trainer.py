import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict
from src.models.siamese import build_siamese_model, contrastive_loss


class SiameseTrainer:
    """Siamese Network 訓練器，支援 Early Stopping 和 checkpoint 儲存"""

    def __init__(self, embedding_dim: int = 256, learning_rate: float = 1e-3):
        self.model = build_siamese_model(embedding_dim=embedding_dim)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=contrastive_loss,
        )

    def train(
        self,
        imgs_a: np.ndarray,
        imgs_b: np.ndarray,
        labels: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        save_dir: str = "models/",
    ) -> Dict:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        use_val = len(imgs_a) >= 10 and validation_split > 0
        callbacks = []

        if use_val:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(save_path / "best_model.h5"),
                    monitor="val_loss",
                    save_best_only=True,
                ),
            ]

        history = self.model.fit(
            x=[imgs_a, imgs_b],
            y=labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split if use_val else 0.0,
            callbacks=callbacks,
            verbose=1,
        )

        self.model.save(str(save_path / "best_model.h5"))
        return history.history

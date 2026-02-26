import numpy as np
import pytest
from src.training.trainer import SiameseTrainer


def _make_dummy_pairs(n=8):
    imgs_a = np.random.rand(n, 224, 224, 3).astype(np.float32)
    imgs_b = np.random.rand(n, 224, 224, 3).astype(np.float32)
    labels = np.array([i % 2 for i in range(n)], dtype=np.float32)
    return imgs_a, imgs_b, labels


def test_trainer_initializes():
    trainer = SiameseTrainer(embedding_dim=64)
    assert trainer.model is not None


def test_trainer_runs_one_epoch(tmp_path):
    trainer = SiameseTrainer(embedding_dim=64)
    imgs_a, imgs_b, labels = _make_dummy_pairs()
    history = trainer.train(
        imgs_a=imgs_a, imgs_b=imgs_b, labels=labels,
        epochs=1, batch_size=4, save_dir=str(tmp_path),
    )
    assert "loss" in history


def test_trainer_saves_model(tmp_path):
    trainer = SiameseTrainer(embedding_dim=64)
    imgs_a, imgs_b, labels = _make_dummy_pairs()
    trainer.train(
        imgs_a=imgs_a, imgs_b=imgs_b, labels=labels,
        epochs=1, batch_size=4, save_dir=str(tmp_path),
    )
    assert len(list(tmp_path.glob("*.h5"))) > 0

import numpy as np
import tensorflow as tf
from src.models.backbone import build_backbone

def test_backbone_output_shape():
    model = build_backbone()
    dummy_input = np.random.rand(2, 224, 224, 3).astype(np.float32)
    output = model(dummy_input, training=False)
    assert len(output.shape) == 2
    assert output.shape[0] == 2

def test_backbone_feature_dim():
    model = build_backbone()
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    output = model(dummy_input, training=False)
    assert output.shape[1] == 1280

def test_backbone_returns_keras_model():
    model = build_backbone()
    assert isinstance(model, tf.keras.Model)

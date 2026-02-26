import numpy as np
import tensorflow as tf
from src.models.siamese import build_siamese_model, contrastive_loss

def test_siamese_output_shape():
    model = build_siamese_model()
    img1 = np.random.rand(2, 224, 224, 3).astype(np.float32)
    img2 = np.random.rand(2, 224, 224, 3).astype(np.float32)
    distances = model([img1, img2], training=False)
    assert distances.shape == (2,)

def test_same_image_zero_distance():
    model = build_siamese_model()
    img = np.random.rand(1, 224, 224, 3).astype(np.float32)
    distances = model([img, img], training=False)
    assert distances.numpy()[0] < 0.01

def test_contrastive_loss_positive_pair():
    y_true = tf.constant([0.0])
    loss_small = contrastive_loss(y_true, tf.constant([0.1]))
    loss_large = contrastive_loss(y_true, tf.constant([0.9]))
    assert loss_small < loss_large

def test_contrastive_loss_negative_pair():
    y_true = tf.constant([1.0])
    loss_small = contrastive_loss(y_true, tf.constant([0.1]))
    loss_large = contrastive_loss(y_true, tf.constant([1.5]))
    assert loss_small > loss_large

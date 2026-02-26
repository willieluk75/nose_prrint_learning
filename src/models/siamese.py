import tensorflow as tf
from src.models.backbone import build_backbone
from src.models.embedding_head import build_embedding_head


def contrastive_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    margin: float = 1.0,
) -> tf.Tensor:
    """
    Contrastive Loss: L = (1-y)*D² + y*max(margin-D, 0)²
    y=0: same pet（距離應小）, y=1: different pet（距離應大）
    """
    y_true = tf.cast(y_true, tf.float32)
    d = tf.cast(y_pred, tf.float32)
    positive_loss = (1.0 - y_true) * tf.square(d)
    negative_loss = y_true * tf.square(tf.maximum(margin - d, 0.0))
    return tf.reduce_mean(positive_loss + negative_loss)


def build_siamese_model(embedding_dim: int = 256) -> tf.keras.Model:
    """
    Siamese Network，共享 backbone + embedding_head 權重。
    輸入兩張圖片，輸出歐氏距離。
    """
    backbone = build_backbone(trainable=False)
    embedding_head = build_embedding_head(input_dim=1280, embedding_dim=embedding_dim)

    input_a = tf.keras.Input(shape=(224, 224, 3), name="image_a")
    input_b = tf.keras.Input(shape=(224, 224, 3), name="image_b")

    emb_a = embedding_head(backbone(input_a))
    emb_b = embedding_head(backbone(input_b))

    distance = tf.keras.layers.Lambda(
        lambda x: tf.norm(x[0] - x[1], axis=1),
        name="euclidean_distance",
    )([emb_a, emb_b])

    return tf.keras.Model(inputs=[input_a, input_b], outputs=distance, name="siamese")

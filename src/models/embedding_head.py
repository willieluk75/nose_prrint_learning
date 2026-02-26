import tensorflow as tf


def build_embedding_head(
    input_dim: int = 1280,
    embedding_dim: int = 256,
) -> tf.keras.Model:
    """
    Embedding projection head，輸出 L2 normalized 向量。
    所有 embedding 在單位超球面上，使歐氏距離與 cosine similarity 計算有效。

    Args:
        input_dim: backbone 輸出維度（MobileNetV2 = 1280）
        embedding_dim: 最終 embedding 維度（預設 256）
    """
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(512, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(embedding_dim)(x)
    x = tf.keras.layers.Lambda(
        lambda v: tf.math.l2_normalize(v, axis=1),
        name="l2_normalize",
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name="embedding_head")

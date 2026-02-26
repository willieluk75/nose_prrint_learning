import tensorflow as tf


def build_backbone(trainable: bool = False) -> tf.keras.Model:
    """
    MobileNetV2 backbone，移除分類頭，輸出 (batch, 1280) 特徵向量。
    使用 ImageNet 預訓練權重。預設凍結所有層（trainable=False）。
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = trainable

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name="backbone")

# model.py
import tensorflow as tf

def conv_block(x, filters, kernel=(3,3), pool=True, dropout=0.0):
    """A reusable convolutional block with BN, ReLU, and optional pooling."""
    x = tf.keras.layers.Conv2D(filters, kernel, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    if pool:
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    if dropout > 0:
        x = tf.keras.layers.SpatialDropout2D(dropout)(x)
    return x

def build_cnn(input_shape, n_classes):
    """Build an improved CNN model for baby cry classification."""
    inputs = tf.keras.Input(shape=input_shape, name="input_layer")

    # Convolutional feature extractor
    x = conv_block(inputs, 32, dropout=0.1)
    x = conv_block(x, 48, dropout=0.1)
    x = conv_block(x, 64, dropout=0.15)
    x = conv_block(x, 96, dropout=0.2)
    x = conv_block(x, 128, dropout=0.25, pool=False)

    # Global pooling + dense layers
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="BabyCryCNN_v2")
    return model

import tensorflow as tf



def vanilla_model(class_count=100):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same'),
        tf.keras.layers.Conv2D(16, (3, 3), padding='same'),
        tf.keras.layers.Conv2D(8, (3, 3), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(class_count, activation="softmax")
    ])
    return model


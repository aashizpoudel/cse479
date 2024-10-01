import tensorflow as tf 


#Design model architecture here.
def model_1(use_regularizer=False):
    # Architecture 1
    if use_regularizer:
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28,28)),
            tf.keras.layers.Dense(128, activation="relu",kernel_regularizer=tf.keras.regularizers.L1()),
            tf.keras.layers.Dense(128, activation="relu",kernel_regularizer=tf.keras.regularizers.L1()),
            tf.keras.layers.Dense(10,activation="softmax")
        ])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28,28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10,activation="softmax")
        ])
    return model


#another model architecture
def model_2(use_regularizer=False, nodes=256):
    if use_regularizer:
        return tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28,28)),
            tf.keras.layers.Dense(nodes, activation="relu",kernel_regularizer=tf.keras.regularizers.L1()),
            tf.keras.layers.Dense(10,activation="softmax")
        ])
    else:
        return tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28,28)),
            tf.keras.layers.Dense(nodes, activation="relu"),
            tf.keras.layers.Dense(10,activation="softmax")
        ])



#another model architecture
def model_3(use_regularizer=False, nodes=128):
    if use_regularizer:
        return tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28,28)),
            tf.keras.layers.Dense(nodes, activation="relu",kernel_regularizer=tf.keras.regularizers.L1()),
            tf.keras.layers.Dense(nodes, activation="relu",kernel_regularizer=tf.keras.regularizers.L1()),
            tf.keras.layers.Dense(nodes, activation="relu",kernel_regularizer=tf.keras.regularizers.L1()),
            tf.keras.layers.Dense(nodes, activation="relu",kernel_regularizer=tf.keras.regularizers.L1()),
            tf.keras.layers.Dense(10,activation="softmax")
        ])
    else:
        return tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28,28)),
            tf.keras.layers.Dense(nodes, activation="relu"),
            tf.keras.layers.Dense(nodes, activation="relu"),
            tf.keras.layers.Dense(nodes, activation="relu"),
            tf.keras.layers.Dense(nodes, activation="relu"),
            tf.keras.layers.Dense(10,activation="softmax")
        ])


def model_4(use_regularizer=False, nodes=128):
    if use_regularizer:
        return tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28,28)),
            tf.keras.layers.Dense(nodes, activation="relu",kernel_regularizer=tf.keras.regularizers.L2()),
            tf.keras.layers.Dense(nodes, activation="relu",kernel_regularizer=tf.keras.regularizers.L2()),
            tf.keras.layers.Dense(10,activation="softmax")
        ])
    else:
        return tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28,28)),
            tf.keras.layers.Dense(nodes, activation="relu"),
            tf.keras.layers.Dense(nodes, activation="relu"),
            tf.keras.layers.Dense(10,activation="softmax")
        ])
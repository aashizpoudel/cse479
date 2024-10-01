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
def model_2():
    pass 


#another model architecture
def model_3():
    pass 


def model_4():
    pass 
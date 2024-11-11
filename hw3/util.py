import re
import string
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
import tensorflow_datasets as tfds 


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

def get_text_vectorizer(dataset, vocab_size=1000, sequence_length=500):
    vectorizer = tf.keras.layers.TextVectorization(standardize=custom_standardization, max_tokens=vocab_size,output_mode='int',output_sequence_length=sequence_length)
    vectorizer.adapt(dataset.map(lambda text,label: text))
    return vectorizer 

def train_model(model, train_ds, val_ds, loss_fn, batch_size=64, optimizer=tf.keras.optimizers.Adam(), callbacks=None, epochs=50):
    train_batched = train_ds.shuffle(10*batch_size).batch(batch_size)
    val_batched = val_ds.shuffle(10*batch_size).batch(batch_size)
    model.compile(loss=loss_fn, optimizer=optimizer,
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)])
    history = model.fit(train_batched, epochs=epochs, validation_data=val_batched,callbacks=callbacks)
    return history

def evaluate_model(model, val_ds, result_path=None):
    val_batched = val_ds.batch(64)
    predictions = (tf.nn.sigmoid(model.predict(val_batched)) > 0.5).numpy()
    truth = np.concatenate([label.numpy() for _,label in val_batched],axis=0)
    accuracy = accuracy_score(truth, predictions)
    ConfusionMatrixDisplay.from_predictions(y_pred=predictions, y_true=truth,normalize="true")
    if result_path:
        plt.savefig(result_path.joinpath('confusion_matrix.jpg'))
        plt.close()
    else:
        plt.show()
    return accuracy 
    
def plot_training_graphs(history, result_path=None):
    plt.plot(history.history['loss'],label="training")
    plt.plot(history.history['val_loss'],label="validation")
    plt.xlabel("epochs")
    if result_path:
        plt.savefig(result_path.joinpath('training_history.jpg'))
        plt.close()
    else:
        plt.show()

def get_train_val_ds():
    DATA_DIR = "./tensorflow-datasets"
    train_ds,val_ds = tfds.load('imdb_reviews', split=["train[:60%]","train[60%:]"], with_info=False,
                          as_supervised=True,data_dir=DATA_DIR)
    return train_ds, val_ds 

def get_for_vocab_ds():
    DATA_DIR = "./tensorflow-datasets"
    return tfds.load("imdb_reviews", split="train", with_info=False, as_supervised=True, data_dir=DATA_DIR)


def get_test_ds():
    DATA_DIR = "./tensorflow-datasets"
    return tfds.load("imdb_reviews", split="test", with_info=False, as_supervised=True, data_dir=DATA_DIR)


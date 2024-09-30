import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np 
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

DATA_DIR = "./tensorflow-datasets"
def get_training_set(batch_size=32):
    training_ds = tfds.load('fashion_mnist', split="train[:80%]",data_dir=DATA_DIR, shuffle_files=False)
    training_ds= training_ds.map(lambda x: (tf.cast(x['image'],tf.float32)/255.0, x['label']))
    training_ds = training_ds.batch(batch_size)
    return training_ds

def get_test_set(batch_size=32):
    ds = tfds.load('fashion_mnist', split="test",data_dir=DATA_DIR, shuffle_files=False)
    ds= ds.map(lambda x: (tf.cast(x['image'],tf.float32)/255.0, x['label']))
    ds = ds.batch(batch_size)
    return ds 

def get_val_set(batch_size=32):
    ds = tfds.load('fashion_mnist', split="train[-20%:]",data_dir=DATA_DIR, shuffle_files=False)
    ds= ds.map(lambda x: (tf.cast(x['image'],tf.float32)/255.0, x['label']))
    ds = ds.batch(batch_size)
    return ds 


def evaluate(model):
    test_ds = get_test_set()
    predictions = model.predict(test_ds).argmax(axis=-1)
    gt = np.concatenate([y for _, y in test_ds], axis=0)
    acc_score = accuracy_score(gt,predictions)
    f1 = f1_score(gt,predictions,average="macro")
    cf = confusion_matrix(gt,predictions)
    return acc_score,f1,cf



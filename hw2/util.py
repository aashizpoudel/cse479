import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.metrics import top_k_accuracy_score,f1_score,confusion_matrix

DATA_DIR = "./tensorflow-datasets"
NUM_CLASSES = 100
def get_train_val_set(train_split="train[:80%]", val_split="train[80%:]"):
    train_ds = tfds.load("cifar100",data_dir=DATA_DIR,split=train_split)
    val_ds = tfds.load("cifar100",data_dir=DATA_DIR,split=val_split)
    return train_ds, val_ds 

    
#Convert data to tuple with image normalized and dataset one hot encoded.
def preprocess_data(data):
    image = tf.cast(data['image'],tf.float32)/255.0 
    label = tf.one_hot(data['label'],NUM_CLASSES)
    return image,label

def augment_data(images,labels):
    # images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    images = tf.image.rot90(images)
    return (images, labels)


def evaluate(model, ds,top_k=1):
    test_ds = ds
    predictions = model.predict(test_ds)
    gt = np.concatenate([tf.math.argmax(y,axis=-1) for _, y in test_ds], axis=0)
    acc_score = top_k_accuracy_score(gt,predictions,k=top_k)
    f1 = f1_score(gt,predictions.argmax(axis=-1),average="macro")
    cf = confusion_matrix(gt,predictions.argmax(axis=-1))
    return acc_score,f1,cf

def evaluate_on_test(model,top_k=1):
    test_ds = tfds.load("cifar100",data_dir=DATA_DIR,split="test")
    test_ds=test_ds.map(preprocess_data).batch(1)
    return evaluate(model,test_ds,top_k=top_k)
    

def get_test_set():
    test_ds = tfds.load("cifar100",data_dir=DATA_DIR,split="test")
    return test_ds

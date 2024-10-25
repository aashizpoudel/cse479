from PIL import Image
import numpy as np
import random
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
import glob
import numpy as np
import tensorflow as tf
import os


def read_npy_file(item):
    data = np.load(item.numpy().decode())
    data = data['arr_0']
    
    data = data.max(axis=-1,keepdims=True)
    return data

def load_data(file_path):
    mask_path = tf.strings.regex_replace(file_path,".jpg",".npz")
    image = tf.io.read_file(file_path)
    image = tf.io.decode_image(image)
    mask = tf.py_function(read_npy_file,[mask_path],tf.uint8)
    return image, mask

# Function to apply padding and random crop for augmentation
def generate_crops_with_padding(image, mask, crop_height=512, crop_width=512):
    image_shape = tf.shape(image)
    img_height, img_width = image_shape[0], image_shape[1]
    new_image_height = crop_height * tf.cast(tf.math.ceil(img_height/crop_height),tf.int32)
    new_image_width = crop_width * tf.cast(tf.math.ceil(img_width/crop_width),tf.int32)
    # Calculate the padding amounts if the image is smaller than the crop size
    pad_height = tf.maximum(new_image_height - img_height, 0)
    pad_width = tf.maximum(new_image_width - img_width, 0)
    
    # Apply padding to the image and mask
    image = tf.image.pad_to_bounding_box(image, 0, 0, img_height + pad_height, img_width + pad_width)
    mask = tf.image.pad_to_bounding_box(mask, 0, 0, img_height + pad_height, img_width + pad_width)
    image_width = new_image_width
    image_height = new_image_height
    # Randomly crop the image and mask to the desired size
    # combined = tf.concat([image, mask], axis=-1)

    
    grid_height = crop_height
    grid_width = crop_width
    num_grids_x = image_width // grid_width
    num_grids_y = image_height // grid_height
    
    # Initialize a list to store the grid images
    grids_i = []
    grids_m = []
  
    # Loop through the image to extract grids
    for y in range(num_grids_y):
        for x in range(num_grids_x):
            # Define the slice window
            y_start = y * grid_height
            x_start = x * grid_width
            i_grid = tf.image.crop_to_bounding_box(
                image, y_start, x_start, grid_height, grid_width
            )
            m_grid = tf.image.crop_to_bounding_box(
                mask, y_start, x_start, grid_height, grid_width
            )
            grids_i.append(i_grid)
            grids_m.append(m_grid)
    images_cropped = tf.stack(grids_i)
    masks_cropped = tf.stack(grids_m,axis=0)

    return images_cropped, masks_cropped

#generate training folders from sampling samples size.
def sample_training_data(txt_file, ds_folder , samples=15, test_split=0.2,crop_height=512,  crop_width=512, directory="/tmp", regenerate=False):
    if Path(f"./tmp/ds_{samples}/train_samples.txt").exists() and not regenerate:
        print("Already generated.")
        return Path(f"./tmp/ds_{samples}")
    
    print("Generating training crops from sample size:", samples)
    training_files = open(txt_file,"r").readlines()
    training_files = [os.path.join(ds_folder,"data",f"{file.strip()}.jpg") for file in training_files]
    
    tr_files, eval_files = train_test_split(training_files, train_size=samples,shuffle=True)
    
    tr_path = f"./tmp/ds_{samples}/train"
    test_path = f"./tmp/ds_{samples}/test"
    Path(tr_path+"/images").mkdir(parents=True, exist_ok=True)
    Path(tr_path+"/masks").mkdir(parents=True, exist_ok=True)
    all_images = []
    all_masks = []
    fp = open(f"./tmp/ds_{samples}/train_samples.txt","w")
    for file in tr_files:
        file_name = file.split(os.sep)[-1].split(".")[0]
        fp.write(file_name+"\r\n")
        image, mask = load_data(file)
        images, masks = generate_crops_with_padding(image,mask, crop_height, crop_width)
        for i in range(len(images)):
            tf.keras.utils.save_img(f"{tr_path}/images/{file_name}_{i+1}.jpg", images[i,:,:,:])
            tf.keras.utils.save_img(f"{tr_path}/masks/{file_name}_{i+1}.png", masks[i,:,:,:])
    fp.close()
    
    fp = open(f"./tmp/ds_{samples}/local_test_samples.txt","w")
    for file in eval_files:
        file_name = file.split(os.sep)[-1].split(".")[0]
        fp.write(file_name+"\r\n")
    fp.close()
    print("Done")
    return Path(f"./tmp/ds_{samples}")

#after sampling


def load_image_and_mask(file_path, masks_folder="masks", mask_extension=".png"):
    mask_path = tf.strings.regex_replace(file_path,"images",masks_folder)
    mask_path = tf.strings.regex_replace(mask_path,".jpg",mask_extension)
    image = tf.io.read_file(file_path)
    image = tf.io.decode_image(image)
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_image(mask)
    return image, mask

def data_augmentation(image,mask):
    image_shape = tf.shape(image)
    img_height, img_width = image_shape[0], image_shape[1]
    
    image =  tf.image.pad_to_bounding_box(image, 1, 1, img_height+10, img_width+10)
    mask =  tf.image.pad_to_bounding_box(mask, 1, 1, img_height+10, img_width+10)
    
    # Randomly crop the image and mask to the desired size
    # image = tf.image.random_brightness(image,max_delta=0.2)
    combined = tf.concat([image, tf.cast(mask,tf.float32)], axis=-1)
    combined_cropped = tf.image.random_crop(combined, (img_height,img_width,4), seed=None, name=None)
    
    combined_cropped = tf.image.random_flip_left_right(combined_cropped)
    combined_cropped = tf.image.rot90(combined_cropped)
    # Separate the image and mask after cropping
    image_cropped = combined_cropped[..., :3]
    mask_cropped = tf.cast(combined_cropped[..., 3:], tf.uint8)
    return image_cropped,mask_cropped

def normalize_data(image,mask):
    image = tf.cast(image,tf.float32)/255.0
    mask = mask/255
    return image,mask


def data_preprocessing(image, mask, augment=False):
    image,mask = normalize_data(image,mask)
    if augment:
        image,mask = data_augmentation(image,mask)
    return image,mask


def load_train_val_ds(ds_path, test_split=0.2, augment_train=True):
    all_files = [str(f) for f in ds_path.glob("train/images/*.jpg")]
    tr, val = train_test_split(all_files, test_size=test_split, shuffle=True)
    t_data = tf.data.Dataset.list_files(tr)
    train_set = t_data.map(load_image_and_mask).map(lambda img,mask: data_preprocessing(img,mask,augment=augment_train))
    val_set = tf.data.Dataset.list_files(val).map(load_image_and_mask).map(lambda img,mask: data_preprocessing(img,mask,augment=False))
    return train_set, val_set 

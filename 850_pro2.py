import numpy
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,Rescaling,RandomZoom
from tensorflow.keras.utils import image_dataset_from_directory


img_height, img_width, channel = 100,100,'rgb'
train_path = './Project 2 Data/Data/Train'
val_path = './Project 2 Data/Data/Validation'
test_path ='./Project 2 Data/Data/Test'

print("importing Data")
train_ds = image_dataset_from_directory(
    train_path,
    seed=501,
    image_size=(img_height, img_width),
    batch_size=32,
    color_mode = channel)

val_ds = image_dataset_from_directory(
    val_path,
   seed=501,
   image_size=(img_height, img_width),
   batch_size=32,
   color_mode = channel)

test_ds = image_dataset_from_directory(
    test_path,
    seed=501,
    image_size=(img_height, img_width),
    batch_size=32,
    color_mode = channel)

preprocs = Sequential([
    Rescaling(1./255)
    #RandomZoom()
    #
    ])



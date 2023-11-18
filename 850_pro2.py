import numpy
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras import layers

from tensorflow.keras.preprocessing.image import random_shear
from tensorflow.keras.utils import image_dataset_from_directory


img_height, img_width, channel = 100,100,'rgb'
shape = (100,100,3)
train_path = './Project 2 Data/Data/Train'
val_path = './Project 2 Data/Data/Validation'
test_path ='./Project 2 Data/Data/Test'

print("importing Data")
train_ds = image_dataset_from_directory(
    train_path,
    label_mode="categorical",
    image_size=(100,100),
    shuffle = True,
    color_mode="rgb",
    seed=501,
    )

val_ds = image_dataset_from_directory(
    val_path,
    label_mode="categorical",
    image_size=(100,100),
    color_mode="rgb",
    shuffle = True,
    seed=501,
    )

val_ds = val_ds.map(lambda x, y: (layers.Rescaling(1.0 / 255)(x), y))

#layers stuff
model = Sequential([
    layers.Rescaling(1.0/255),
    layers.RandomZoom(0.1),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    
    layers.Conv2D(16, 6, activation= 'relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 6, activation= 'relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, activation= 'relu'),
    layers.Conv2D(64, 3, activation= 'relu'),
    layers.Conv2D(64, 3, activation= 'relu'),
    layers.MaxPooling2D(),
    
    
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax'),
])



model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

m = model.fit(
    train_ds,
    validation_data= val_ds,
    epochs = 10)


acc = m.history['accuracy']
val_acc = m.history['val_accuracy']
loss = m.history['loss']
val_loss = m.history['val_loss']

import matplotlib.pyplot as plt

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()



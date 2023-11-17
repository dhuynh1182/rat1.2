import numpy
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D,MaxPooling2D,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


img_height, img_width, channel = 100,100,'rgb'
shape = (100,100,3)
train_path = './Project 2 Data/Data/Train'
val_path = './Project 2 Data/Data/Validation'
test_path ='./Project 2 Data/Data/Test'

# print("importing Data")
# train_ds = image_dataset_from_directory(
#     train_path,
#     seed=501,
#     image_size=(img_height, img_width),
#     batch_size=32,
#     color_mode = channel)

# val_ds = image_dataset_from_directory(
#     val_path,
#     seed=501,
#     image_size=(img_height, img_width),
#     batch_size=32,
#     color_mode = channel)

# test_ds = image_dataset_from_directory(
#     test_path,
#     seed=501,
#     image_size=(img_height, img_width),
#     batch_size=32,
#     color_mode = channel)

#figure this out later
train_gen = ImageDataGenerator(
    shear_range = 0.2,
    zoom_range = 0.2,
    rescale = 1./255 )

valid_gen = ImageDataGenerator(
    rescale =1./255 
    )


train_ds = train_gen.flow_from_directory(
        train_path,  # this is the target directory
        target_size=(100, 100),
        color_mode = channel,
        batch_size=32,
        class_mode='binary')

val_ds = valid_gen.flow_from_directory(
    val_path,
    target_size= (100,100),
    color_mode = channel,
    batch_size = 32,
    class_mode = 'binary'
    )

#layers stuff
# model = Sequential()
# model.layer
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=shape))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(4))

model = Sequential()
model.add(Conv2D(16, (11,11), input_shape=shape,activation = 'relu'))
model.add(MaxPooling2D())

model.add(Conv2D(32, (5, 5), input_shape=shape,activation = 'relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3, 3), input_shape=shape,activation = 'relu'))
model.add(MaxPooling2D())


model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(4))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_ds,
    validation_data= val_ds,
    epochs = 10)




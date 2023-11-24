import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np 
from keras import layers

img_height, img_width, channel = 100,100,'rgb'
shape = (100,100,3)
train_path = './Project 2 Data/Data/Train'
val_path = './Project 2 Data/Data/Validation'
test_path ='./Project 2 Data/Data/Test'

print("importing Data")

train_datagen  = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(
    rescale = 1./255,
    )

# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        train_path,  # this is the target directory
        target_size=(img_height,img_width),     
        batch_size=32,
        class_mode='categorical',
        color_mode='rgb',
        shuffle = True
)

# batches of augmented image data
val_generator = val_datagen.flow_from_directory(
        val_path,  # this is the target directory
        target_size=(img_height,img_width),  
        batch_size=32,
        class_mode='categorical',
        color_mode='rgb',
        shuffle = True
)

print(train_generator.class_indices)


#layers stuff
model = Sequential([  
    
    layers.Conv2D(128, 3, activation= 'relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, activation= 'leaky_relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.5),
    
    layers.Conv2D(64, 3, activation= 'relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    
    
    layers.Flatten(),
    layers.Dense(32), 
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax'),
])



model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


m = model.fit(
    train_generator,
    validation_data=val_generator ,
    epochs = 60) 

model.summary()

model.save("model_D")

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

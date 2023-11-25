from tensorflow.keras.utils import load_img,img_to_array
from tensorflow.keras.models import load_model
import numpy as np


from warnings import filterwarnings
from tensorflow import io
from tensorflow import image
from matplotlib import pyplot as plt



def image_loader(path):
    
    shape = [100,100]
    load = load_img(path, target_size = shape)
    input_arr = img_to_array(load)
    input_arr = np.array([input_arr])
    
    
    return input_arr

    


path1 = 'Project 2 Data\Data\Test\Medium\Crack__20180419_06_19_09,915.bmp'
path2 = 'Project 2 Data\Data\Test\Large\Crack__20180419_13_29_14,846.bmp'


img1 = image_loader(path1)/255      
img2 = image_loader(path2)/255


model = load_model("model_D")

pred_img1= model.predict(img1)
pred_img2 = model.predict(img2)


filterwarnings("ignore") 
fig, ax = plt.subplots()


#FOR MED
tf_img = io.read_file(path1)
tf_img = image.decode_png(tf_img, channels=3)
fig = plt.imshow(tf_img)
plt.title("True Crack Class: Medium")

# Define crack classes
crack_classes = {'Large': 0, 'Medium': 1, 'None': 2, 'Small': 3}

# Plot percentage text for each class
for label, index in crack_classes.items():
    value = pred_img1[0,index]*100
    ax.text(1150, 1450 + index*100, f"{label}: {value: .2f}%", c = 'green')


# Turn off axes for a cleaner display
ax.axis('off')

# Show the plot
plt.show()
   

#FOR LARGE 
fig, ax = plt.subplots()


tf_img = io.read_file(path2)
tf_img = image.decode_png(tf_img, channels=3)
fig = plt.imshow(tf_img)
plt.title("True Crack Class: Large")

# Define crack classes
crack_classes = {'Large': 0, 'Medium': 1, 'None': 2, 'Small': 3}

# Plot percentage text for each class
for label, index in crack_classes.items():
    value = pred_img2[0,index]*100
    ax.text(1150, 1450 + index*100, f"{label}: {value: .2f}%", c = 'green')


# Turn off axes for a cleaner display
ax.axis('off')

# Show the plot
plt.show()
   

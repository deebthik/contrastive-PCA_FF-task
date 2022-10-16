import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

target_idx = np.where(mnist.train.labels<2)[0]
foreground = mnist.train.images[target_idx,:][:2000]
target_labels = mnist.train.labels[target_idx][:2000]

tf.logging.set_verbosity(old_v)


from PIL import Image
from object_detection.utils import resize_and_crop
import os

IMAGE_PATH = 'Replace with your own path to downloaded images' 
        
natural_images = list() #dictionary of pictures indexed by the pic # and each value is 100x100 image
for filename in os.listdir(IMAGE_PATH):
    if filename.endswith(".JPEG") or filename.endswith(".JPG") or filename.endswith(".jpg"):
        try:
            im = Image.open(os.path.join(IMAGE_PATH,filename))
            im = im.convert(mode="L") #convert to grayscale
            im = resize_and_crop(im) #resize and crop each picture to be 100px by 100px
            natural_images.append(np.reshape(im, [10000])) 
        except Exception as e:
            pass #print(e)
            
natural_images=np.asarray(natural_images,dtype=float)
natural_images/=255 #rescale to be 0-1
print("Array of grass images:",natural_images.shape)

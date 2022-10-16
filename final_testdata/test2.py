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


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
fg = pca.fit_transform(foreground)
colors = ['k','r']

for i, l in enumerate(np.sort(np.unique(target_labels))):
    plt.scatter(fg[np.where(target_labels==l),0],fg[np.where(target_labels==l),1], 
                color=colors[i], label='Digit ' +str(l))
plt.legend()

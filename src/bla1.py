
import numpy as np
from contrastive import CPCA

'''
N = 500; D = 30; gap=3
# In B, all the data pts are from the same distribution, which has different variances in three subspaces.
B = np.zeros((N, D))
B[:,0:10] = np.random.normal(0,10,(N,10))
B[:,10:20] = np.random.normal(0,3,(N,10))
B[:,20:30] = np.random.normal(0,1,(N,10))
'''

'''
# In A there are four clusters.
A = np.zeros((N, D))
A[:,0:10] = np.random.normal(0,10,(N,10))
# group 1
A[0:100, 10:20] = np.random.normal(0,1,(100,10))
A[0:100, 20:30] = np.random.normal(0,1,(100,10))
# group 2
A[100:200, 10:20] = np.random.normal(0,1,(100,10))
A[100:200, 20:30] = np.random.normal(gap,1,(100,10))
# group 3
A[200:300, 10:20] = np.random.normal(2*gap,1,(100,10))
A[200:300, 20:30] = np.random.normal(0,1,(100,10))
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import os
import numpy

A=[]
i = 0
directory = os.fsencode("/Users/deebthik/Desktop/test_FF/final_testdata/final_superimposed_images_6000")

for file in os.listdir(directory):
    #Opening Image and resizing to 10X10 for easy viewing
    image_test = np.array(Image.open('/Users/deebthik/Desktop/test_FF/final_testdata/final_superimposed_images_6000/' + os.fsdecode(file)).resize((28,28)).convert('L'))  #note: I used a local image
    #print image
    #print (image_test)

    #manipulate the array
    x=np.array(image_test)
    #convert to 1D vector
    y=np.concatenate(x)
    #print (len(y))
    A += [y]
    
    i += 1
    print ("A iteration - " + str(i))
    if i == 5000:
        break
    

A_final = numpy.array(A)
#print (A_final.shape)




B=[]
i = 0
directory = os.fsencode("/Users/deebthik/Desktop/test_FF/final_testdata/satellite_images")

for file in reversed(os.listdir(directory)):

    #if i <= 2500:
        #i += 1
        #print ("B iteration skipped - " + str(i))
        #continue
    
    #Opening Image and resizing to 10X10 for easy viewing
    image_test = np.array(Image.open('/Users/deebthik/Desktop/test_FF/final_testdata/satellite_images/' + os.fsdecode(file)).resize((28,28)).convert('L'))  #note: I used a local image
    #print image
    #print (image_test)

    #manipulate the array
    x=np.array(image_test)
    #convert to 1D vector
    y=np.concatenate(x)
    #print (len(y))
    B += [y]
    
    i += 1
    print ("B iteration - " + str(i))

    if i == 5000:
        break
    
    

B_final = numpy.array(B)
#print (B_final.shape)


#A_labels = [0]*200+[1]*200+[2]*200+[3]*200+[4]*200
A_labels = [0]*300+[1]*300+[2]*300

cpca = CPCA(standardize=False)
cpca.fit_transform(A_final, B_final, plot=True, active_labels=A_labels)




'''
from contrastive import CPCA
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from PIL import Image
import numpy


# Loading the image 
img = cv2.imread('/Users/deebthik/Desktop/test_FF/final_testdata/final_superimposed_images_6000/6 copy.png',0) #loading in grayscale
#plt.imshow(img)
np_img = numpy.array(img)
print(np_img.shape)
print(np_img.data)


img2 = cv2.imread('/Users/deebthik/Desktop/test_FF/final_testdata/satellite_images/0a0f81b.jpg',0) #loading in grayscale
img2 = cv2.resize(img2, (64, 64))
#plt.imshow(img2)
np_img2 = numpy.array(img2)
print(np_img2.shape)
print(np_img2.data)


cpca = CPCA(standardize=False)
cpca.fit_transform(np_img.data, np_img2.data, plot=True)
'''






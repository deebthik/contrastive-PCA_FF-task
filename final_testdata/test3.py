
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
    image_test = numpy.array(Image.open('/Users/deebthik/Desktop/test_FF/final_testdata/final_superimposed_images_6000/' + os.fsdecode(file)).resize((25,25)).convert('L'))  #note: I used a local image
    #print image
    #print (image_test)

    #manipulate the array
    x=numpy.array(image_test)
    #convert to 1D vector
    y=numpy.concatenate(x)
    #print (len(y))
    A += [y]
    
    i += 1
    print ("A iteration - " + str(i))
    if i == 1000:
        break
    

A_final = numpy.array(A)
#print (A_final.shape)

#Import required modules
from sklearn.decomposition import PCA
 
pca = PCA(2) # we need 2 principal components.
converted_data = pca.fit_transform(A_final)


plt.style.use('seaborn-whitegrid')
plt.figure(figsize = (10,6))
c_map = plt.cm.get_cmap('jet', 10)
plt.scatter(converted_data[:, 0], converted_data[:, 1], s = 15,
            cmap = c_map)
plt.colorbar()
plt.xlabel('PC-1') , plt.ylabel('PC-2')
plt.show()




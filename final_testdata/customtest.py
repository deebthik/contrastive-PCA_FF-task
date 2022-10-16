'''
# Importing required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from PIL import Image
import numpy


# Loading the image 
#img = cv2.imread('/Users/deebthik/Desktop/test_FF/final_testdata/final_superimposed_images_6000/grayscale.png') #you can use any image you want.
img = cv2.imread('/Users/deebthik/Desktop/test_FF/final_testdata/final_superimposed_images_6000/6 copy.png',0) #loading in grayscale
plt.imshow(img)
np_img = numpy.array(img)
print(np_img.shape)
print(np_img.data)

data = np_img.data

pca = PCA(2) # we need 2 principal components.
converted_data = pca.fit_transform(np_img.data)
 
converted_data.shape

plt.style.use('seaborn-whitegrid')
plt.figure(figsize = (10,6))
c_map = plt.cm.get_cmap('jet', 10)
plt.scatter(converted_data[:, 0], converted_data[:, 1], s = 15,
            cmap = c_map)
plt.colorbar()
plt.xlabel('PC-1') , plt.ylabel('PC-2')
plt.show()
'''



'''
import cv2
import os
import numpy as np
from PIL import Image

i = 0
new_data = []
directory = os.fsencode("/Users/deebthik/Desktop/test_FF/final_testdata/satellite_images")

for file in os.listdir(directory):
    myFile = "/Users/deebthik/Desktop/test_FF/final_testdata/satellite_images/" + os.fsdecode(file)
    image = cv2.imread(myFile)
    img = cv2.resize(image , (8 , 8)) # Reshaping the testing images to 32*32
    #new_data.append(img)
    new_data.append(np.array(Image.open(myFile,'r')).flatten())
    i += 1
    print ("iteration - " + str(i))
    if i == 50:
        break


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca=PCA(2)
converted_data = pca.fit(new_data)

plt.style.use('seaborn-whitegrid')
plt.figure(figsize = (10,6))
c_map = plt.cm.get_cmap('jet', 10)
plt.scatter(converted_data[:, 0], converted_data[:, 1], s = 15,
            cmap = c_map) #, c = digits.target)
plt.colorbar()
plt.xlabel('PC-1') , plt.ylabel('PC-2')
plt.show()


#new_data = np.reshape(new_data, (len(new_data), 1024))
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

#Opening Image and resizing to 10X10 for easy viewing
image_test = np.array(Image.open('/Users/deebthik/Desktop/test_FF/final_testdata/final_superimposed_images_6000/1.png').resize((16,16)).convert('L'))  #note: I used a local image
#print image
print (image_test)

#manipulate the array
x=np.array(image_test)
#convert to 1D vector
y=np.concatenate(x)
print (len(y))

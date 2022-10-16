from PIL import Image

import numpy as np
import os, random

'''
img = Image.open("/Users/deebthik/Desktop/test_FF/final_testdata/arabicletters_four/id_2558_label_19.png").convert("RGBA")



size = (256, 256)
img = img.resize(size, Image.ANTIALIAS)


background = Image.open("/Users/deebthik/Desktop/test_FF/final_testdata/arabicletters_four/0a0f81b.jpg").convert("RGBA")


background = background.resize(size,Image.ANTIALIAS)

background.paste(img, (0, 0), img)
background.save('/Users/deebthik/Desktop/how_to_superimpose_two_images_02.png',"PNG")
'''




i = 1
directory = os.fsencode("/Users/deebthik/Desktop/test_FF/final_testdata/satellite_images")
    
for file in os.listdir(directory):
    
    filename = "/Users/deebthik/Desktop/test_FF/final_testdata/satellite_images/" + os.fsdecode(file)

    letter_img_rand = "/Users/deebthik/Desktop/test_FF/final_testdata/arabicletters_four/all_letters_final/" + random.choice(os.listdir("/Users/deebthik/Desktop/test_FF/final_testdata/arabicletters_four/all_letters_final"))

    img = Image.open(letter_img_rand).convert("RGBA")
    size = (28, 28)
    #img = img.resize(size, Image.ANTIALIAS)
    img = img.resize(size)

    background = Image.open(filename).convert("RGBA")
    background = background.resize(size,Image.ANTIALIAS)
    background.paste(img, (0, 0), img)
    background.save('/Users/deebthik/Desktop/test_FF/final_testdata/final_superimposed_images_6000/' + str(i) + ".png","PNG")

    print ("Iteration - " + str(i))
    i += 1

    #if i == 5000:
        #break

    
    

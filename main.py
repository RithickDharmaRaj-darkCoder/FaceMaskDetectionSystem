import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn.model_selection import train_test_split


with_mask_files = os.listdir('Dataset\with_mask')
without_mask_files = os.listdir('Dataset\without_mask')

# Creating labels for each images as 0's and 1's....
with_mask_labels = [1] * len(with_mask_files)
without_mask_labels = [0] * len(without_mask_files)
labels = with_mask_labels + without_mask_labels

img = mpimg.imread('Dataset\without_mask\\0_0_anhu_0056.jpg')
imgplot = plt.imshow(img)
plt.show()


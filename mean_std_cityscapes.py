""" 
Author: Sorour Mohajerani--smohajer@sfu.ca
Date: Nov 2019

- This code generates the depth wise mean and std of images in the cityscapes training set. A csv file which
contains all the desired images have been already saved in the 'leftImg8bit' folder.

- Usually having a csv file, which contains desired filenames in it, makes it easier to do a specific task on
a number of images. If you decided to exclude some of those images or to use only some specific ones, just 
apply the changes in the csv file and run the same code without changing it.

- scv fils for train, val, and test can be found here:

- For calculating channel wise std over a dataset, we need to have all of the arrays of images in one place.
""" 


from __future__ import print_function
import os
import numpy as np
from skimage.io import imread
from tqdm import tqdm
import pandas as pd

root = 'PATH TO THE CITYSCAPES DIRECTORY\leftImg8bit'
csv_filename = 'train_images.csv'
desired_channel_folder = 'train'  # could be train/test/val

csv_filpath = os.path.join(root, csv_filename)
image_folder_path = os.path.join(root, desired_channel_folder)

df_train_img = pd.read_csv(csv_filpath) # if need to read the first m filenames of the csv file: pd.read_csv(csv_filpath, nrows=m)
all_img = []
for ix, filenamest in enumerate(tqdm(df_train_img['name'], miniters=1000)):
    tmp = []
    city = filenamest.split('_')[0]
    nimage = city + '/' + filenamest + '.png'
    tmp = imread(os.path.join(image_folder_path, nimage))
    tmp = np.array(tmp)
    #tmp = np.resize(tmp, (256, 512, 3)) # if do not have enough memory, reduce the size
    tmp = np.reshape(tmp, (-1, 3))
    all_img.append(tmp)

all_img = np.array(all_img)
print('shape of the entire data', all_img.shape)

ch1 = np.reshape(all_img[:, 0], (-1, 1)) / 255.
ch2 = np.reshape(all_img[:, 1], (-1, 1)) / 255.
ch3 = np.reshape(all_img[:, 2], (-1, 1)) / 255.

mean1 = np.mean(ch1)
mean2 = np.mean(ch2)
mean3 = np.mean(ch3)
data_mean = np.array([mean1, mean2, mean3])

std1 = np.std(ch1)
std2 = np.std(ch2)
std3 = np.std(ch3)
data_std = np.array([std1, std2, std3])

print('mean:', data_mean, '\n std:', data_std, '\n # of images inevstigated: ', ix + 1)

# shape of the entire data (2975, 2097152, 3)
# mean: [0.29866842 0.30135223 0.30561872]
# std: [0.23925215 0.23859318 0.2385942 ]
# # of images inevstigated:  2975

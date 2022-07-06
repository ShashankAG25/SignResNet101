import pickle
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import cv2
# Bounding_Box = namedtuple('Bounding_Box', 'xmin ymin xmax ymax')
# data_pros =np.load("data_pros.npy",allow_pickle=True)
# #for checking lets print 6 of them
# for _ in range(6):
#     i = np.random.randint(len(data_pros))
#     image,bounding_box = data_pros[i]
#     print(bounding_box)
#     plt.imshow(image)
#     plt.show()


x_train = np.load("x_train.npy")
x_validation = np.load("x_validation.npy")
y_box_train= np.load("y_box_train.npy")
y_box_validation = np.load("y_box_validation.npy")
y_class_train = np.load("y_class_train.npy")
y_class_validation = np.load("y_class_validation.npy")
print(x_train.shape,x_validation.shape,y_box_train.shape,y_box_validation.shape,y_class_train.shape,y_class_validation.shape)
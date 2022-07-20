import pickle
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
# Bounding_Box = namedtuple('Bounding_Box', 'xmin ymin xmax ymax')
# data_pros =np.load("data_pros.npy",allow_pickle=True)
# #for checking lets print 6 of them
# for _ in range(6):
#     i = np.random.randint(len(data_pros))
#     image,bounding_box = data_pros[i]
#     print(bounding_box)
#     plt.imshow(image)
#     plt.show()

#
# x_train = np.load("x_train.npy")
# x_validation = np.load("x_validation.npy")
# y_box_train= np.load("y_box_train.npy")
# y_box_validation = np.load("y_box_validation.npy")
# y_class_train = np.load("y_class_train.npy")
# y_class_validation = np.load("y_class_validation.npy")
# print(x_train.shape,x_validation.shape,y_box_train.shape,y_box_validation.shape,y_class_train.shape,y_class_validation.shape
Bounding_Box = namedtuple('Bounding_Box', 'xmin ymin xmax ymax')

data_pros = np.load("data.npy", allow_pickle=True)

def plot_with_box(image_data, bounding_box, compare_box=None):
    fig,ax = plt.subplots(1)
    ax.imshow(image_data)
    # Creating a Rectangle patch for the changed one
    boxA = patches.Rectangle((bounding_box.xmin, bounding_box.ymin),
    bounding_box.xmax - bounding_box.xmin,
    bounding_box.ymax - bounding_box.ymin,
    linewidth=3, edgecolor='y', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(boxA)
    #Creating another Rectangular patch for the real one
    if compare_box is not None:
        boxB = patches.Rectangle((compare_box.xmin, compare_box.ymin),
        compare_box.xmax - compare_box.xmin,
        compare_box.ymax - compare_box.ymin,
        linewidth=2, edgecolor='b', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(boxB)
    plt.show()





i = np.random.randint(len(data_pros))
image, bounding_box = data_pros[i]
plot_with_box(image, bounding_box)



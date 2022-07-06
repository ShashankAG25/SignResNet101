from collections import namedtuple
import csv
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from collections import namedtuple
import cv2
from PIL import Image, ImageOps
import xml.etree.ElementTree as ET

Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

# data pathes
data_images = "SPODS/img/"
# data_ClassList = '/content/drive/MyDrive/AI_dataset_pets/annotations/list.txt'
data_xmlAnnotations = "SPODS/labels_xml/"
TARGET_SIZE = (224, 224)

# BoundingBox
# The following function will read the xml and return the values for xmin, ymin, xmax, ymax for formulating the bounding box
Bounding_Box = namedtuple('Bounding_Box', 'xmin ymin xmax ymax')


def get_bounding_box(path_to_xml_annotation):
    tree = ET.parse(path_to_xml_annotation)
    root = tree.getroot()
    path_to_box = './object/bndbox/'
    xmin = int(root.find(path_to_box + "xmin").text)
    ymin = int(root.find(path_to_box + "ymin").text)
    xmax = int(root.find(path_to_box + "xmax").text)
    ymax = int(root.find(path_to_box + "ymax").text)
    return Bounding_Box(xmin, ymin, xmax, ymax)


# padding for making the image to a perfect square and apply the needed changes to the bounding box according to the padding and rescaling.
def resize_image_with_bounds(path_to_image, bounding_box=None, target_size=None):
    image = Image.open(path_to_image)
    width, height = image.size
    w_pad = 0
    h_pad = 0
    bonus_h_pad = 0
    bonus_w_pad = 0
    # the following code helps determining where to pad or is it not necessary for the images we have.
    # If the difference between the width and height was odd((height<width)case), we add one pixel on one side
    # If the difference between the height and width was odd((height>width)case), then we add one pixel on one side.
    # if both of these are not the case, then pads=0, no padding is needed, since the image is already a square itself.
    if width > height:
        pix_diff = (width - height)
        h_pad = pix_diff // 2
        bonus_h_pad = pix_diff % 2
    elif height > width:
        pix_diff = (height - width)
        w_pad = pix_diff // 2
        bonus_w_pad = pix_diff % 2
    # When we pad the image to square, we need to adjust all the bounding box values by the amounts we added on the left or top.
    # The "bonus" pads are always done on the bottom and right so we can ignore them in terms of the box.
    image = ImageOps.expand(image, (w_pad, h_pad, w_pad + bonus_w_pad, h_pad + bonus_h_pad))
    if bounding_box is not None:
        new_xmin = bounding_box.xmin + w_pad
        new_xmax = bounding_box.xmax + w_pad
        new_ymin = bounding_box.ymin + h_pad
        new_ymax = bounding_box.ymax + h_pad
    # We need to also apply the scalr to the bounding box which we used in resizing the image
    if target_size is not None:
        # So, width and height have changed due to the padding resize.
        width, height = image.size
        image = image.resize(target_size)
        width_scale = target_size[0] / width
        height_scale = target_size[1] / height
    if bounding_box is not None:
        new_xmin = new_xmin * width_scale
        new_xmax = new_xmax * width_scale
        new_ymin = new_ymin * height_scale
        new_ymax = new_ymax * height_scale
    image_data = np.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
    # The image data is a 3D array such that 3 channels ,RGB of target_size.(RGB values are 0-255)
    if bounding_box is None:
        return image_data, None
    return (image_data, Bounding_Box(new_xmin, new_ymin, new_xmax, new_ymax))


def setting_sample_from_name(sample_name):
    path_to_image = os.path.join(data_images, sample_name + '.png')
    path_to_xml = os.path.join(data_xmlAnnotations, sample_name + '.xml')
    original_bounding_box = get_bounding_box(path_to_xml)
    image_data, bounding_box = resize_image_with_bounds(path_to_image, original_bounding_box, TARGET_SIZE)
    return (image_data, bounding_box)




# def plot_with_box(image_data, bounding_box, compare_box=None):
#     # compare_box = bounding_box
#     fig, ax = plt.subplots(1)
#     ax.imshow(image_data)
#     # Creating a Rectangle patch for the changed one
#     boxA = patches.Rectangle((bounding_box.xmin, bounding_box.ymin),
#                              bounding_box.xmax - bounding_box.xmin,
#                              bounding_box.ymax - bounding_box.ymin,
#                              linewidth=3, edgecolor='y', facecolor='none')
#     # Add the patch to the Axes
#     ax.add_patch(boxA)
#     # Creating another Rectangular patch for the real one
#     if compare_box is not None:
#         boxB = patches.Rectangle((compare_box.xmin, compare_box.ymin),
#                                  compare_box.xmax - compare_box.xmin,
#                                  compare_box.ymax - compare_box.ymin,
#                                  linewidth=2, edgecolor='b', facecolor='none')
#         # Add the patch to the Axes
#         ax.add_patch(boxB)
#     # FOR FINDING INTERSECTION OVER UNION
#     xA = max(bounding_box.xmin, compare_box.xmin)
#     yA = max(bounding_box.ymin, compare_box.ymin)
#     xB = min(bounding_box.xmax, compare_box.xmax)
#     yB = max(bounding_box.ymax, compare_box.ymax)
#     interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
#     boxAArea = (bounding_box.xmax - bounding_box.xmin + 1) * (bounding_box.ymax - bounding_box.ymin + 1)
#     boxBArea = (compare_box.xmax - compare_box.xmin + 1) * (compare_box.ymax - compare_box.ymin + 1)
#     iou = interArea / float(boxAArea + boxBArea - interArea)
#     # By intersection of union I mean intersection over union(IOU) #itself
#     print('intersection of union =', iou)
#     plt.show()

#test abouve code
# sample_name = '11'
# image, bounding_box = setting_sample_from_name(sample_name)
# plot_with_box(image, bounding_box)


#creating np array of preprocessed data


data_pros = []
# class_id = 0
for i in(os.listdir(data_images)):
    filename = i.split(".")[0]
    print(filename)
    image, bounding_box = setting_sample_from_name(filename)
        # print(bounding_box)
        # plt.imshow(image)
        # plt.show()
    data_tuple = (image, bounding_box)
    data_pros.append(data_tuple)
# print(data_pros)
# print(f'Processed {len(data_pros)} samples')
data_pros = np.array(data_pros)
# np.save("data_pros", data_pros)


#Tran and test splits
x_train = []
y_class_train = []
y_box_train = []
x_validation = []
y_class_validation = []
y_box_validation = []
validation_split = 0.2
for image,bounding_box in data_pros:
    if np.random.random() > validation_split:
        x_train.append(preprocess_input(image))
        y_class_train.append(0)
        y_box_train.append(bounding_box)
    else:
        x_validation.append(preprocess_input(image))
        y_class_validation.append(1)
        y_box_validation.append(bounding_box)
x_train = np.array(x_train)
y_class_train = np.array(y_class_train)
y_box_train = np.array(y_box_train)
x_validation = np.array(x_validation)
y_class_validation = np.array(y_class_validation)
box_validation =np.array(y_box_validation)

#saving the np array
np.save("x_train",x_train)
np.save("y_class_train",y_class_train)
np.save("y_box_train",y_box_train)
np.save("x_validation",x_validation)
np.save("y_class_validation",y_class_validation)
np.save("y_box_validation",y_box_validation)

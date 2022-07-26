##  source    :::::  https://medium.com/@muskulpesent/sliding-windows-for-object-detection-with-python-709250eb6161

from PIL import Image
import numpy as np
import cv2
import os, cv2, keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = "SPODS/img/"
annot = "SPODS/labels_csv/"

def create_regions(image):
    image = image
    regions = []
    stepSize = 112
    (w_width, w_height) = (stepSize,stepSize )
    for x in range(0, image.shape[1] - w_width, stepSize):
        for y in range(0, image.shape[0] - w_height, stepSize):
            regions.append([x, y, (x + w_width), (y + w_height)])
    return regions

train_images = []
train_labels = []
#

def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

#
#
# for e, i in enumerate(os.listdir(annot)):
#         filename = i.split(".")[0] + ".png"
#         print(e, filename)
#         image = cv2.imread(os.path.join(path, filename))
#         df = pd.read_csv(os.path.join(annot, i))
#         gtvalues = []
#         for row in df.iterrows():
#             x1 = int(row[1][0].split(" ")[0])
#             y1 = int(row[1][0].split(" ")[1])
#             x2 = int(row[1][0].split(" ")[2])
#             y2 = int(row[1][0].split(" ")[3])
#             cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255), 2)
#             gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
#         regions = create_regions(image)
#         imout = image.copy()
#         counter = 0
#         falsecounter = 0
#         flag = 0
#         fflag = 0
#         bflag = 0
#         for e, result in enumerate(regions):
#             if e < 2000 and flag == 0:
#                 for gtval in gtvalues:
#                     x, y, w, h = result
#                     iou = get_iou(gtval, {"x1": x, "x2": w, "y1": y, "y2":h})
#                     print(iou)
#                     if counter < 30:
#                         if iou > 0.1:
#                             timage = imout[y:h, x:w]
#                             resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
#                             train_images.append(resized)
#                             train_labels.append(1)
#                             counter += 1
#
#                     else:
#                         fflag = 1
#                     if falsecounter < 30:
#                         if iou < 0.1:
#                             timage = imout[y:h, x:w]
#                             resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
#                             train_images.append(resized)
#                             train_labels.append(0)
#                             falsecounter += 1
#                     else:
#                         bflag = 1
#                 if fflag == 1 and bflag == 1:
#                     print("inside")
#                     flag = 1
#
# X_new = np.array(train_images)
# y_new = np.array(train_labels)
#
# np.save("x_new",X_new)
# np.save("y_new",y_new)
#
image = cv2.imread("SPODS/img/image (23).png")
region = create_regions(image)

for rects in region:
    x,y,w,h = rects
    cv2.rectangle(image, (x, y), (w, h), (255, 0, 0), 2)  # draw rectangle on image
cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
cv2.imshow("Output", image)
cv2.waitKey(0)




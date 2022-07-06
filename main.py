#  source
# https://towardsdatascience.com/object-localization-using-pre-trained-cnn-models-such-as-mobilenet-resnet-xception-f8a5f6a0228d
# https://github.com/1297rohit/RCNN/blob/master/RCNN.ipynb

import os, cv2, keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

path = "sign_crop/"
# annot = "sign/label/"

train_images=[]
train_labels = []

for e, i in enumerate(os.listdir(path)):
        img = cv2.imread(os.path.join(path,i))
        resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        train_images.append(resized)
        train_labels.append(1)
        # df = pd.read_csv(os.path.join(annot, i))
        # gtvalues = []
        # for row in df.iterrows():
        #     x1 = int(row[1][0].split(" ")[0])
        #     y1 = int(row[1][0].split(" ")[1])
        #     x2 = int(row[1][0].split(" ")[2])
        #     y2 = int(row[1][0].split(" ")[3])
        #     gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
        # ss.setBaseImage(image)
        # ss.switchToSelectiveSearchFast()
        # # ssresults = ss.process()
        # imout = image.copy()
        # for gtval in gtvalues:
        #             x, y, w, h = gtval.values()
        #             timage = imout[y:y + h, x:x + w]
        #             # resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
        #             train_images.append(timagem12
        #                                 )
        #             train_labels.append(1)
        #             flag = 1

X_new = np.array(train_images)
y_new = np.array(train_labels)

np.save("x_new",X_new)
np.save("y_new",y_new)
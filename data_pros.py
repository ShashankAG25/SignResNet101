#resizing with box ::::  https://stackoverflow.com/questions/49466033/resizing-image-and-its-bounding-box
# boudning box training:::  https://pyimagesearch.com/2020/10/05/object-detection-bounding-box-regression-with-keras-tensorflow-and-deep-learning/

import os

from keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

BASE_PATH = "SPODS"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "img/"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "labels_csv"])
print(IMAGES_PATH)
print(ANNOTS_PATH)

# BASE_OUTPUT = "output"
# MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
# PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
# TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])

INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 2

print("[INFO] loading dataset...")
# rows = open(ANNOTS_PATH).read().strip().split("\n")
data = []
targets = []
filenames = []

for e,i in enumerate(os.listdir(ANNOTS_PATH)):
        filename = i.split(".")[0]+".png"
        print(filename)
        img = cv2.imread(os.path.join(IMAGES_PATH,filename))
        df = pd.read_csv(os.path.join(ANNOTS_PATH,i))
        for row in df.iterrows():
            x1 = int(row[1][0].split(" ")[0])
            y1 = int(row[1][0].split(" ")[1])
            x2 = int(row[1][0].split(" ")[2])
            y2 = int(row[1][0].split(" ")[3])
        # image =cv2.resize(img, (224, 224))
        startX = int(x1)
        startY = int(y1)
        # Also initialize ending point
        endX = int(x2)
        endY = int(y2)

        y_ = img.shape[0]
        x_ = img.shape[1]
        targetSize = 1080
        x_scale = targetSize / x_
        y_scale = targetSize / y_
        # print(x_scale, y_scale)
        image = cv2.resize(img, (targetSize, targetSize));
        # print(image.shape)
        x = int(np.round(startX * x_scale))
        y = int(np.round(startY * y_scale))
        xmax = int(np.round(endX* x_scale))
        ymax = int(np.round(endY * y_scale))
        image = img_to_array(image)
        data.append(image)
        targets.append((x, y, xmax, ymax))
        filenames.append(filename)

# print(data)
# print(targets.shape)
# print(len(filenames))

# # Normalizing Data here also we face would face issues if we take input as integer

data=np.array(data,dtype='float32') / 255.0
targets=np.array(targets,dtype='float32')
split=train_test_split(data,targets,filenames,test_size=0.10,random_state=42)
(train_images,test_images) = split[:2]
(train_targets,test_targets) = split[2:4]
(train_filenames,test_filenames) = split[4:]

vgg=ResNet50(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))
vgg.summary()

# we use VGG16 as per our requirement not use whole
vgg.trainable = False

flatten = vgg.output

flatten = Flatten()(flatten)

# Lets make bboxhead
bboxhead = Dense(128,activation="relu")(flatten)
bboxhead = Dense(64,activation="relu")(bboxhead)
bboxhead = Dense(32,activation="relu")(bboxhead)
bboxhead = Dense(4,activation="relu")(bboxhead)

model = Model(inputs = vgg.input,outputs = bboxhead)
model.summary()

opt = Adam(1e-4)

model.compile(loss='mse',optimizer=opt)

checkpoint = ModelCheckpoint("Resnet.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

history = model.fit(train_images,train_targets,validation_data=(test_images,test_targets),steps_per_epoch= 10, epochs= 300, validation_steps=2, callbacks=[checkpoint])



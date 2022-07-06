from collections import namedtuple
import csv
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
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
import numpy as np
import cv2

Detection = namedtuple("Detection", ["image_path", "gt", "pred"])
from PIL import Image, ImageOps
from tensorflow.keras.optimizers import Adam
import xml.etree.ElementTree as ET
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint, EarlyStopping

# x_train = np.load("x_train.npy")
# x_validation = np.load("x_validation.npy")
# y_box_train= np.load("y_box_train.npy")
# y_box_validation = np.load("y_box_validation.npy")
# y_class_train = np.load("y_class_train.npy")
# y_class_validation = np.load("y_class_validation.npy")


# import X_new and y_new

X_new = np.load("x_new.npy")
y_new = np.load("y_new.npy")

print(len(X_new))


ResNetModel = ResNet50(weights='imagenet', include_top=True)
ResNetModel.summary()

for layers in (ResNetModel.layers)[:49]:
    print(layers)
    layers.trainable = False

X = ResNetModel.layers[-2].output
predictions = Dense(2, activation="softmax")(X)

model_final = Model(ResNetModel.input, predictions)

opt = Adam(lr=0.0001)
model_final.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])
model_final.summary()


class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1 - Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)


lenc = MyLabelBinarizer()
Y = lenc.fit_transform(y_new
                       )
X_train, X_test , y_train, y_test = train_test_split(X_new,Y,test_size=0.10)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#
trdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
traindata = trdata.flow(x=X_train, y=y_train)
tsdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
testdata = tsdata.flow(x=X_test, y=y_test)

checkpoint = ModelCheckpoint("ResNetModel.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

hist = model_final.fit_generator(generator= traindata, steps_per_epoch=10, epochs= 300, validation_data= testdata, validation_steps=2, callbacks=[checkpoint])


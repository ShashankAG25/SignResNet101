import os
import cv2
from tensorflow.keras.preprocessing.image import load_img
# we also save images into array format so import img_array library too
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Input,Flatten,Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

base_path = "SPODS"
images = os.path.sep.join([base_path, 'img'])
annotations = os.path.sep.join([base_path, 'labels.csv'])

# print(images)
# print(annotations)

# Lets Load Dataset
# airplanes annotation is a Csv file thats why we can see through with rows

rows = open(annotations).read().strip().split("\n")

# lets make three list where we save our exact bounding boxes
data = []
targets = []
filenames = []
# print(rows)

# After load we have to split dataset according to images
# import some usefull libraries
for row in rows:
    row = row.split(",")
    # we always create rectangle with h+w so we have to know where exactly we should start from
    (classname, startX, startY, endX, endY, filename, res_width, res_height) = row

    imagepaths = os.path.sep.join([images, filename])
    image = cv2.imread(imagepaths)
    img = image.copy()
    (h, w) = image.shape[:2]

    # initializing starting point
    # Why we take in float because when we convert into array so then will trouble happen
    startX = float(startX) / w
    startY = float(startY) / h
    # Also initialize ending point
    endX = float(endX) / w
    endY = float(endY) / h

    cv2.rectangle(img, (startX,startY), (endX,endY), (255, 0, 0), 2)

    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    cv2.imshow("Output" ,img)
    cv2.waitKey(0)
    # load image and give them default size
    image = load_img(imagepaths, target_size=(224, 224))
    # see here if we cant take it into float then we face trouble
    image = img_to_array(image)

    # Lets append into data , targets ,filenames
    targets.append((startX, startY, endX, endY))
    filenames.append(filename)
    data.append(image)
#
# # Normalizing Data here also we face would face issues if we take input as integer
#
# data=np.array(data,dtype='float32') / 255.0
# targets=np.array(targets,dtype='float32')
# split=train_test_split(data,targets,filenames,test_size=0.10,random_state=42)
# (train_images,test_images) = split[:2]
# (train_targets,test_targets) = split[2:4]
# (train_filenames,test_filenames) = split[4:]
#
# vgg=ResNet50(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))
# vgg.summary()
#
# # we use VGG16 as per our requirement not use whole
# vgg.trainable = False
#
# flatten = vgg.output
#
# flatten = Flatten()(flatten)
#
# # Lets make bboxhead
# bboxhead = Dense(128,activation="relu")(flatten)
# bboxhead = Dense(64,activation="relu")(bboxhead)
# bboxhead = Dense(32,activation="relu")(bboxhead)
# bboxhead = Dense(4,activation="relu")(bboxhead)
#
# model = Model(inputs = vgg.input,outputs = bboxhead)
# model.summary()
#
# opt = Adam(1e-4)
#
# model.compile(loss='mse',optimizer=opt)
#
# checkpoint = ModelCheckpoint("Resnet.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#
# history = model.fit(train_images,train_targets,validation_data=(test_images,test_targets),steps_per_epoch= 10, epochs= 300, validation_steps=2, callbacks=[checkpoint])
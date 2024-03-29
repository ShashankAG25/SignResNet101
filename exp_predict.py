
# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2

print("[INFO] loading object detector...")
# model = load_model("output/detector.h5")
model = load_model("resnet.h5")
# loop over the images that we'll be testing using our bounding box
# regression model
image = load_img("SPODS/img/image (333).png", target_size=(224, 224))
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)
# make bounding box predictions on the input image
preds = model.predict(image)[0]
(startX, startY, endX, endY) = preds
# load the input image (in OpenCV format), resize it such that it
# fits on our screen, and grab its dimensions
image = cv2.imread("SPODS/img/image (333).png")
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]
# scale the predicted bounding box coordinates based on the image
# dimensions
startX = int(startX * w)
startY = int(startY * h)
endX = int(endX * w)
endY = int(endY * h)
# draw the predicted bounding box on the image
cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 2)
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)






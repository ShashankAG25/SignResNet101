import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im



x = np.load("x_new.npy")
y = np.load("y_new.npy")

# print(y)

for i in y:
    # print(i)
    if i==1:
        img = x[i]
        plt.imshow(img)
        plt.show()
import cv2
import numpy as np
from matplotlib import pyplot as plt


def grayAndBlur(image, gaussianKernelSize):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (gaussianKernelSize, gaussianKernelSize), 0)


def otsuize(
    image,
    lowOtsuThresh: int = 0,
    highOtsuThresh: int = 255,
):
    return cv2.threshold(
        image, lowOtsuThresh, highOtsuThresh, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]


row, col = 1, 2
fig, axs = plt.subplots(row, col)
fig.tight_layout()

image = cv2.imread("input.jpg")

gb = grayAndBlur(image, 3)
thresh = otsuize(gb)

cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
    # ROI = original[y : y + h, x : x + w]
    # cv2.imwrite("ROI_{}.png".format(ROI_number), ROI)
    # ROI_number += 1

axs[0].imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
axs[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

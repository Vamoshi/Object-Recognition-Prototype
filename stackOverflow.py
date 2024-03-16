# THIS IS FOR EXPERIMENTAL PURPOSES AND IS THUS VERY CLUTTERED CHECK objectDetection.py
import cv2
import numpy as np
from matplotlib import pyplot as plt

row, col = 1, 3
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

# Load image, grayscale, Otsu's threshold
image = cv2.imread("input.png")
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

axs[0].imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))

# Find contours, obtain bounding box, extract and save ROI
ROI_number = 0
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
    ROI = original[y : y + h, x : x + w]
    # cv2.imwrite("ROI_{}.png".format(ROI_number), ROI)
    ROI_number += 1

cv2.imshow("image", image)
cv2.waitKey()

# NOTES TO SELF: playing around with variables gaussianKernelSize, kernelSize, and threshold changes a lot.
# THIS IS FOR EXPERIMENTAL PURPOSES AND IS THUS VERY CLUTTERED CHECK sample.py

import cv2
import numpy as np
from matplotlib import pyplot as plt

row, col = 1, 5
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

image = cv2.imread("input.png")

# Make Grayscale because... it helps
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

gaussianKernelSize = 1
blurred = cv2.GaussianBlur(gray, (gaussianKernelSize, gaussianKernelSize), 0)

binImage = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

axs[1].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
axs[2].imshow(cv2.cvtColor(binImage, cv2.COLOR_BGR2RGB))

# Dilation to close up lines
morphKernelSize = 3
kernel = np.ones((morphKernelSize, morphKernelSize), np.uint8)
morph = None
# morph = cv2.dilate(binImage, kernel, iterations=1)

if morph is not None:
    # morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    # morph = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)
    # morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    pass
else:
    morph = cv2.morphologyEx(binImage, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(binImage, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(binImage, cv2.MORPH_GRADIENT, kernel)
    pass


axs[3].imshow(cv2.cvtColor(morph, cv2.COLOR_BGR2RGB))

# Find contours then sort them from biggest to smallest contour area
# Courtesy of StackOverflow - idk what contours are
contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
contours = sorted(contours, key=cv2.contourArea, reverse=True)
# print(contours)

mergedContours = []
# bounding box combination threshold
threshold = 300
# # Merge Contours
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    current_rect = (x, y, x + w, y + h)

    merged = False
    for i, mergedRect in enumerate(mergedContours):
        if (
            abs(x - mergedRect[0]) < threshold
            and abs(y - mergedRect[1]) < threshold
            and abs(x + w - mergedRect[2]) < threshold
            and abs(y + h - mergedRect[3]) < threshold
        ):
            # Merge the bounding rectangles
            mergedContours[i] = (
                min(x, mergedRect[0]),
                min(y, mergedRect[1]),
                max(x + w, mergedRect[2]),
                max(y + h, mergedRect[3]),
            )
            merged = True
            break

    if not merged:
        mergedContours.append(current_rect)

## create a copy of the image to draw the bounding boxes onto - just so i know where the bounding boxes are
imageCopy = image.copy()

# mergedContours[1:] because the first bounding box is usually the entire image

## Draw bounding boxes for merged contours
for rect in mergedContours[1:]:
    x, y, x2, y2 = rect
    cv2.rectangle(imageCopy, (x, y), (x2, y2), (36, 255, 12), 2)

# ## Save Images for merged
# for i, rect in enumerate(mergedContours):
#     if i == 0:
#         continue
#     x, y, x2, y2 = rect
#     roi = image[y:y2, x:x2]
#     cv2.imwrite(f"ingredient{i}.png", roi)


# ## For non-merged contours
# #Dont recommend using this one
# boxSizeOffset = 1
# # change list ending to desired number of objects, empty means all of them
# for i, contour in enumerate(contours[1:]):
#     x, y, width, height = cv2.boundingRect(contour)
#     cv2.rectangle(
#         imageCopy,
#         (x, y),
#         (x + int(width * boxSizeOffset), y + int(height * boxSizeOffset)),
#         (36, 255, 12),
#         2,
#     )
#     roi = imageCopy[y : y + height, x : x + width]
#     # cv2.imwrite(f"object{i}.png", roi)


axs[4].imshow(cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB))

# Show the plot
plt.show()

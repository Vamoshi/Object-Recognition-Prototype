import cv2
import numpy as np
from matplotlib import pyplot as plt


def saveImages(mergedContours):
    for i, rect in enumerate(mergedContours):
        if i == 0:
            continue
        x, y, x2, y2 = rect
        roi = image[y:y2, x:x2]
        cv2.imwrite(f"ingredient{i}.png", roi)


def grayAndBlur(image, gaussianKernelSize):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (gaussianKernelSize, gaussianKernelSize), 0)


def otsuize(image, lowOtsuThresh, highOtsuThresh):
    return cv2.threshold(
        image, lowOtsuThresh, highOtsuThresh, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]


def mergeContours(contours, mergingThresh):
    mergedContours = []
    # # Merge Contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        current_rect = (x, y, x + w, y + h)

        merged = False
        for i, merged_rect in enumerate(mergedContours):
            if (
                abs(x - merged_rect[0]) < mergingThresh
                and abs(y - merged_rect[1]) < mergingThresh
                and abs(x + w - merged_rect[2]) < mergingThresh
                and abs(y + h - merged_rect[3]) < mergingThresh
            ):
                # Merge the bounding rectangles
                mergedContours[i] = (
                    min(x, merged_rect[0]),
                    min(y, merged_rect[1]),
                    max(x + w, merged_rect[2]),
                    max(y + h, merged_rect[3]),
                )
                merged = True
                break

        if not merged:
            mergedContours.append(current_rect)

    return mergedContours


def detectImages(
    image,
    gaussianKernelSize: int = 1,
    morphKernelSize: int = 3,
    lowOtsuThresh: int = 0,
    highOtsuThresh: int = 255,
    mergingThresh: int = 300,
):
    blurred = grayAndBlur(image, gaussianKernelSize)
    binImage = otsuize(blurred, lowOtsuThresh, highOtsuThresh)
    morphKernel = np.ones((morphKernelSize, morphKernelSize), np.uint8)
    morph = None

    # I'm experimenting with different morphological ops... need advice which ones are the best - too many combinations
    # morph = cv2.dilate(binImage, morphKernel, iterations=1)
    ## Uncommenting morph above goes in the if below
    if morph is not None:
        # morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, morphKernel)
        # morph = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, morphKernel)
        # morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, morphKernel)
        # morph = cv2.morphologyEx(binImage, cv2.MORPH_TOPHAT, morphKernel)
        pass
    else:
        # morph = cv2.morphologyEx(binImage, cv2.MORPH_CLOSE, morphKernel)
        morph = cv2.morphologyEx(binImage, cv2.MORPH_GRADIENT, morphKernel)
        morph = cv2.morphologyEx(binImage, cv2.MORPH_TOPHAT, morphKernel)
        # morph = cv2.morphologyEx(binImage, cv2.MORPH_OPEN, morphKernel)
        # morph = cv2.morphologyEx(binImage, cv2.MORPH_BLACKHAT, morphKernel)
        pass

    if morph is None:
        morph = binImage

    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return mergeContours(contours, mergingThresh)


row, col = 1, 2
fig, axs = plt.subplots(row, col)
fig.tight_layout()

image = cv2.imread("input.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

## create a copy of the image to draw the bounding boxes onto - just so i know where the bounding boxes are
imageCopy = image.copy()

mergedcontours = detectImages(
    image,
    mergingThresh=300,
    # Seems morphKernelSize shouldn't be an extreme number, range between 3-33 looks good
    # I will probably change this to a list of objects morphOp:KernelSize to make it more customizable
    morphKernelSize=3,
    # after experimenting, it seems increasing the gaussian kernel size reduces bounding box accuracy of encapsulating individual objects
    # Which makes sense since if the image is too blurry, then everything melds into one
    gaussianKernelSize=1,
)

## Draw bounding boxes for merged contours
for rect in mergedcontours[1:]:
    x, y, x2, y2 = rect
    cv2.rectangle(imageCopy, (x, y), (x2, y2), (36, 255, 12), 2)

axs[0].imshow(cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB))


plt.show()

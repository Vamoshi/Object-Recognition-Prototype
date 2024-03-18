import cv2
import numpy as np
from matplotlib import pyplot as plt


class MorphOperation:
    def __init__(self, operation, kernelSize, iterations=1):
        self.operation = operation
        self.kernel = np.ones((kernelSize, kernelSize), np.uint8)
        self.iterations = iterations


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
    morphOps=[],
    gaussianKernelSize: int = 1,
    lowOtsuThresh: int = 0,
    highOtsuThresh: int = 255,
    mergingThresh: int = 300,
):
    blurred = grayAndBlur(image, gaussianKernelSize)
    binImage = otsuize(blurred, lowOtsuThresh, highOtsuThresh)
    morph = binImage

    # I'm experimenting with different morphological ops... need advice which ones are the best - too many combinations
    for i, morphOp in enumerate(morphOps):
        morph = cv2.morphologyEx(
            morph, morphOp.operation, morphOp.kernel, iterations=morphOp.iterations
        )

    # Retr_external doesn't work good
    contours = cv2.findContours(morph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Return MergedContours, blurredImage, and morphed image
    return (mergeContours(contours, mergingThresh), blurred, morph)


row, col = 1, 4
fig, axs = plt.subplots(row, col, figsize=(30, 10))

# fig.tight_layout()

image = cv2.imread("input.png")

## create a copy of the image to draw the bounding boxes onto - just so i know where the bounding boxes are
imageCopy = image.copy()

morphOps = [
    # MorphOperation(cv2.MORPH_ERODE, 3),
    # MorphOperation(cv2.MORPH_DILATE, 7),
    MorphOperation(cv2.MORPH_CLOSE, 3, 1),
    MorphOperation(cv2.MORPH_CLOSE, 3, 1),
    MorphOperation(cv2.MORPH_OPEN, 3),
    # MorphOperation(cv2.MORPH_GRADIENT, 3),
    # MorphOperation(cv2.MORPH_TOPHAT, 3),
    # MorphOperation(cv2.MORPH_BLACKHAT, 3),
]

mergedcontours, blurred, morph = detectImages(
    image,
    morphOps=morphOps,
    mergingThresh=300,
    # after experimenting, it seems increasing the gaussian kernel size reduces bounding box accuracy of encapsulating individual objects
    # Which makes sense since if the image is too blurry, then everything melds into one
    gaussianKernelSize=3,
)

## Draw bounding boxes for merged contours
for rect in mergedcontours[1:]:
    x, y, x2, y2 = rect
    cv2.rectangle(imageCopy, (x, y), (x2, y2), (36, 255, 12), 2)

axs[0].set_title("image")
axs[1].set_title("blurred")
axs[2].set_title("morph")
axs[3].set_title("final")

axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[1].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
axs[2].imshow(cv2.cvtColor(morph, cv2.COLOR_BGR2RGB))
axs[3].imshow(cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB))

saveImages(mergedcontours)

plt.show()

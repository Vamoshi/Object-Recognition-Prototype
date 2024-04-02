import base64
import gzip
import io
import json
import cv2
from matplotlib import pyplot as plt
import numpy as np
from classes.MorphOperation import MorphOperation
from classes.MetaClasses import NonInstantiableMeta


class ImageMethod(metaclass=NonInstantiableMeta):

    def grayAndBlur(image, gaussianKernelSize):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (gaussianKernelSize, gaussianKernelSize), 0)

    def otsuize(image, lowOtsuThresh=0, highOtsuThresh=255):
        return cv2.threshold(
            image, lowOtsuThresh, highOtsuThresh, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

    def sobelize(image, kernelSize):
        sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernelSize)
        sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernelSize)

        return cv2.convertScaleAbs(cv2.magnitude(sobelX, sobelY))

    def saveDetectedObjects(image, mergedContours):
        for i, rect in enumerate(mergedContours):
            if i == 0:
                continue
            x, y, x2, y2 = rect
            roi = image[y:y2, x:x2]
            cv2.imwrite(f"ingredient{i}.png", roi)

    def saveImages(images):
        for i, imageBinary in enumerate(images):

            # Convert the binary image to a NumPy array
            numpyImage = np.frombuffer(imageBinary, dtype=np.uint8)

            # Decode the NumPy array as an image using OpenCV
            image = cv2.imdecode(numpyImage, cv2.IMREAD_UNCHANGED)

            cv2.imwrite(f"image{i}.png", image)
        print("Image Saved")

    def makeThumbnail(image, targetSize):
        h, w = image.shape[:2]
        aspectRatio = w / h

        newH = newW = targetSize

        if aspectRatio > 1:
            newW = targetSize
            newH = int(newW / aspectRatio)
        else:
            newH = targetSize
            newW = int(newH * aspectRatio)

        return cv2.resize(image, (newW, newH))

    def encodeForReturn(imageList):
        returnImages = []

        for i, img in enumerate(imageList):
            cv2EncodedImage = cv2.imencode(".png", img)[1]
            base64Encode = base64.b64encode(cv2EncodedImage)
            decoded = base64Encode.decode()
            returnImages.append(decoded)
        return returnImages


# def getContours(
#     image,
#     morphOps=[],
#     gaussianKernelSize: int = 1,
#     lowOtsuThresh: int = 0,
#     highOtsuThresh: int = 255,
# ):
#     blurred = grayAndBlur(image, gaussianKernelSize)
#     sobelled = cv2.convertScaleAbs(sobelize(blurred, 3))
#     binImage = otsuize(sobelled, lowOtsuThresh, highOtsuThresh)
#     morph = binImage

#     # I'm experimenting with different morphological ops... need advice which ones are the best - too many combinations
#     for i, morphOp in enumerate(morphOps):
#         morph = cv2.morphologyEx(
#             morph, morphOp.operation, morphOp.kernel, iterations=morphOp.iterations
#         )

#     # Retr_external doesn't work good
#     contours = cv2.findContours(morph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)

#     # Return contours, blurredImage, and morphed image
#     return (contours, blurred, morph)


# row, col = 1, 5
# fig, axs = plt.subplots(row, col, figsize=(16, 9))

# # fig.tight_layout()

# image = cv2.imread("input2.jpg")

# morphOps = [
#     # MorphOperation(cv2.MORPH_ERODE, 3),
#     # MorphOperation(cv2.MORPH_DILATE, 7),
#     MorphOperation(cv2.MORPH_CLOSE, 3, 3),
#     # MorphOperation(cv2.MORPH_CLOSE, 3, 1),
#     # MorphOperation(cv2.MORPH_OPEN, 3),
#     # MorphOperation(cv2.MORPH_GRADIENT, 3),
#     # MorphOperation(cv2.MORPH_TOPHAT, 3),
#     # MorphOperation(cv2.MORPH_BLACKHAT, 3),
# ]

# # Get Contours
# contours, blurred, morph = getContours(
#     image,
#     morphOps=morphOps,
#     # after experimenting, it seems increasing the gaussian kernel size reduces bounding box accuracy of encapsulating individual objects
#     # Which makes sense since if the image is too blurry, then everything melds into one
#     gaussianKernelSize=15,
# )

# # Then mergeContours
# # TODO: change merge contour algorithm return format to contours not plain list of points
# mergedContours = contours

# # mergedContours = mergeContoursTopLeftDist(mergedContours, 40)
# # mergedContours = mergeContours(mergedContours, 300)
# mergedContours = mergeContoursCenterDist(mergedContours, 200)

# ## create a copy of the image to draw the bounding boxes onto - just so i know where the bounding boxes are
# imageCopy = image.copy()
# imageCopy2 = image.copy()

# sobelled = cv2.convertScaleAbs(sobelize(grayAndBlur(imageCopy2, 3), 3))

# ## Draw bounding boxes for merged contours
# for rect in mergedContours:
#     x, y, x2, y2 = rect
#     cv2.rectangle(imageCopy, (x, y), (x2, y2), (36, 255, 12), 2)

# axs[0].set_title("image")
# axs[1].set_title("blurred")
# axs[2].set_title("sobel")
# axs[3].set_title("morph")
# axs[4].set_title("final")

# axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# axs[1].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
# axs[2].imshow(cv2.cvtColor(sobelled, cv2.COLOR_BGR2RGB))
# axs[3].imshow(cv2.cvtColor(morph, cv2.COLOR_BGR2RGB))
# axs[4].imshow(cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB))

# saveImages(mergedContours)
# print(len(mergedContours))
# print(type(image))

# plt.show()

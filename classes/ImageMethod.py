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

    # Prepare image for object detection
    def grayAndBlur(image, gaussianKernelSize):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (gaussianKernelSize, gaussianKernelSize), 0)

    # Apply Otsu's thresholding
    def otsuize(image, lowOtsuThresh=0, highOtsuThresh=255):
        return cv2.threshold(
            image, lowOtsuThresh, highOtsuThresh, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

    # Apply Sobel thresholding
    def sobelize(image, kernelSize):
        sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernelSize)
        sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernelSize)
        return cv2.convertScaleAbs(cv2.magnitude(sobelX, sobelY))

    # Saves list of images to local
    def saveImages(images):
        for i, imageBinary in enumerate(images):

            # Convert the binary image to a NumPy array
            numpyImage = np.frombuffer(imageBinary, dtype=np.uint8)

            # Decode the NumPy array as an image using OpenCV
            image = cv2.imdecode(numpyImage, cv2.IMREAD_UNCHANGED)

            cv2.imwrite(f"image{i}.png", image)
        print("Image Saved")

    # Compress image by reducing its size
    # Works but i dont recommend
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

    # Converts CV2 images to base64
    def encodeForReturn(imageList):
        returnImages = []

        for i, img in enumerate(imageList):
            cv2EncodedImage = cv2.imencode(".png", img)[1]
            base64Encode = base64.b64encode(cv2EncodedImage)
            decoded = base64Encode.decode()
            returnImages.append(decoded)
        return returnImages

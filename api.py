import base64
import io
import json
import time
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from classes.ContourMerge import ContourMerge
from classes.ImageMethod import ImageMethod
from PIL import Image

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate(r"../firebaseKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

app = Flask(__name__)

baseRoute = "/api"


@app.route(f"{baseRoute}/", methods=["POST", "GET"])
def index():
    print("IN BASE PATH")
    return "Welcome to the Image Processing API!"


@app.route(f"{baseRoute}/upload_image", methods=["POST"])
def uploadImage():
    # Receive the binary image data from the request
    binaryImageData = request.get_data()

    print(base64.b64encode(cv2.imencode(".png", binaryImageData)[1]).decode())

    # Return a response indicating successful upload
    return jsonify({"message": "Image received"})


@app.route(f"{baseRoute}/test_firebase", methods=["POST"])
def testFirebase():
    # Receive the binary image data from the request
    # binaryImageData = request.get_data()

    # print(base64.b64encode(cv2.imencode(".png", binaryImageData)[1]).decode())

    if not request.form["image"]:
        return jsonify({"message": "No image"})

    base64Image = request.form["image"]

    data = {"unprocessed": base64Image}

    docRef = db.collection("tempDetectedImagesCollection").document()
    docRef.set(data)

    print(docRef.id)

    return jsonify({"message": docRef.id})


@app.route(f"{baseRoute}/detect_objects", methods=["POST"])
def detectObjects():
    # try:
    if not request.form["image"]:
        return jsonify({"message": "No image"})

    base64Image = request.form["image"]

    decodedBytes = base64.b64decode(base64Image)
    # Convert bytes to numpy array
    buffer_array = np.frombuffer(decodedBytes, dtype=np.uint8)
    # Decode the image from buffer using OpenCV
    image = cv2.imdecode(buffer_array, cv2.IMREAD_COLOR)

    imageCopy = image.copy()
    image = ImageMethod.grayAndBlur(image, 1)
    image = ImageMethod.sobelize(image, 1)
    image = ImageMethod.otsuize(image)
    contours = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    contours = ContourMerge.mergeContoursGenDist(contours, 300)

    # # Draw bounding boxes for merged contours
    # for rect in contours:
    #     x, y, x2, y2 = rect
    #     cv2.rectangle(imageCopy, (x, y), (x2, y2), (36, 255, 12), 2)

    # print(len(contours))
    detectedObjects = []

    for i, rect in enumerate(contours):
        if i == 0:
            continue
        x, y, x2, y2 = rect
        roi = imageCopy[y:y2, x:x2]
        detectedObjects.append(roi)
        print("doing Stuff")

    # cv2.imshow("Decoded Image", imageCopy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # thumbnails = [cv2.resize(img, (50, 50)) for img in detectedObjects]

    # cv2.imencode econdes the images into a numpy array
    # base64.b64encode encodes it into base64
    # decode, decodes the base64 into a string

    # print(base64.b64encode(cv2.imencode(".png", detectedObjects[0])[1]).decode)

    # returnImages = [
    #     jsonify({i: base64.b64encode(cv2.imencode(".png", img)[1]).decode()})
    #     for i, img in enumerate(detectedObjects)
    # ]

    # thumbnails = [ImageMethod.makeThumbnail(img, 100) for img in detectedObjects]
    # returnImages = ImageMethod.encodeForReturn(thumbnails)

    # for i, img in enumerate(thumbnails):
    #     cv2EncodedImage = cv2.imencode(".png", img)[1]
    #     base64Encode = base64.b64encode(cv2EncodedImage)
    #     decoded = base64Encode.decode()
    #     # time.sleep(0.3)
    #     returnImages.append(decoded)

    # imageCopy

    # with open("output.txt", "w") as f:
    #     f.write(returnImages[0])
    #     # for string in returnImages:
    #     #     print("OUTPUTTING")
    #     #     f.write(string + "\n \n ")

    returnImages = ImageMethod.encodeForReturn(detectedObjects)

    parentCollection = db.collection("tempImages")
    parentDoc = parentCollection.document()
    childCollection = parentDoc.collection("images")

    for image in returnImages:
        childCollection.add({image: image})
        # print(docRef.id)

    # data = {"detectedObjects": returnImages}
    # docRef.set(data)

    return json.dumps(
        {
            "message": "Image received",
            "detectedObjects": parentDoc.id,
        }
    )

    # returnImages = ["hello"]

    # return jsonify({"message": "Image received", "returnImages": returnImages})


# except Exception as e:
#     return jsonify({"message": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

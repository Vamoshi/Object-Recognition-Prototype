import base64
import json
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from classes.ContourMerge import ContourMerge
from classes.ImageMethod import ImageMethod

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate(r"../firebaseKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "exp://10.50.78.104:8081"}})


baseRoute = "/api"


@app.route(f"{baseRoute}/", methods=["POST", "GET"])
def index():
    return "Welcome to the Image Processing API!"


# TODO: Implement feature to allow front end to specify MorphOperations & parameters
@app.route(f"{baseRoute}/detect_objects", methods=["POST"])
def detectObjects():
    if not request.form or len(request.form) < 1:
        return jsonify({"message": "No Form"})

    if not request.form["image"]:
        return jsonify({"message": "No image"})

    base64Image = request.form["image"]

    # Encode received image to CV2-acceptable format
    decodedBytes = base64.b64decode(base64Image)
    # Convert bytes to numpy array
    buffer_array = np.frombuffer(decodedBytes, dtype=np.uint8)
    # Decode the image from buffer using OpenCV
    image = cv2.imdecode(buffer_array, cv2.IMREAD_COLOR)

    # Create a copy of the image because the original image is going to be modified
    imageCopy = image.copy()

    # Modify original image
    image = ImageMethod.grayAndBlur(image, 1)
    image = ImageMethod.sobelize(image, 1)
    image = ImageMethod.otsuize(image)
    # TODO: add morphops here

    # Find Contours
    contours = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]
    # Sort from biggest to smallest image
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # MergeContours using ContourMerge functions
    contours = ContourMerge.mergeContoursGenDist(contours, 300)

    # Store detected images
    detectedObjects = []
    for i, rect in enumerate(contours):
        # skip the first one because it's usually a very general detection
        if i == 0:
            continue
        x, y, x2, y2 = rect
        roi = imageCopy[y:y2, x:x2]
        detectedObjects.append(roi)

    # Encode detected Objects into base64
    returnImages = ImageMethod.encodeForReturn(detectedObjects)

    # Access the correct Google Firestore doc
    parentCollection = db.collection("tempImages")
    parentDoc = parentCollection.document()
    childCollection = parentDoc.collection("images")

    # Add to Google Firestore
    for im in returnImages:
        childCollection.add({"image": im})

    return json.dumps(
        {
            "message": "Image received",
            "documentID": parentDoc.id,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

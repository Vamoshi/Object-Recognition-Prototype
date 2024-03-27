import base64
import io
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from classes.ContourMerge import ContourMerge
from classes.ImageMethod import ImageMethod
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "exp://192.168.1.106:8081"}})

baseRoute = "/api"


@app.route(f"{baseRoute}/", methods=["POST", "GET"])
def index():
    print("IN BASE PATH")
    return "Welcome to the Image Processing API!"


@app.route(f"{baseRoute}/upload_image", methods=["POST"])
def uploadImage():
    # Receive the binary image data from the request
    binaryImageData = request.get_data()

    print(binaryImageData)

    # Return a response indicating successful upload
    return jsonify({"message": "Image received"})


@app.route(f"{baseRoute}/detect_objects", methods=["POST"])
def detectObjects():
    # try:
    base64Image = request.form.to_dict()["image"]
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

    contours = ContourMerge.mergeContoursGenDist(contours, 400)

    ## Draw bounding boxes for merged contours
    # for rect in contours:
    #     x, y, x2, y2 = rect
    #     cv2.rectangle(imageCopy, (x, y), (x2, y2), (36, 255, 12), 2)

    print(len(contours))
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

    # cv2.imencode econdes the images into a numpy array
    # base64.b64encode encodes it into base64
    # decode, decodes the base64 into a string
    returnImages = [
        base64.b64encode(cv2.imencode(".png", img)[1]).decode()
        for img in detectedObjects
    ]

    print(len(returnImages))

    return jsonify(
        {
            "message": "Image received",
            "detectedObjects": returnImages,
        }
    )


# except Exception as e:
#     return jsonify({"message": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

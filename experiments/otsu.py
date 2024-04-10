# THIS IS FOR EXPERIMENTAL PURPOSES AND IS THUS VERY CLUTTERED CHECK sample.py
import cv2
import numpy as np
from matplotlib import pyplot as plt

row, col = 1, 5
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

image = cv2.imread("input.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Apply Gaussian blur and Otsu's thresholding
size = 15
blurred = cv2.GaussianBlur(gray, (size, size), 0)
binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

axs[1].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
axs[2].imshow(cv2.cvtColor(binary_image, cv2.COLOR_BGR2RGB))

# Dilate the binary image
kernel = np.ones((3, 3), np.uint8)
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

axs[3].imshow(cv2.cvtColor(dilated_image, cv2.COLOR_BGR2RGB))

# Find contours and sort by area
contours = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
contours = sorted(contours, key=cv2.contourArea, reverse=True)

boxSizeOffset = 2

# Extract and save ROIs
for i, contour in enumerate(contours[:]):  # Change 3 to the desired number of objects
    x, y, width, height = cv2.boundingRect(contour)
    cv2.rectangle(
        image,
        (x, y),
        (x + int(width * boxSizeOffset), y + int(height * boxSizeOffset)),
        (36, 255, 12),
        2,
    )
    roi = image[y : y + height, x : x + width]
    cv2.imwrite(f"object{i}.png", roi)

print(f"{len(contours)} objects extracted and saved.")
axs[4].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

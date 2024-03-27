import cv2
from matplotlib import pyplot as plt
from classes.MorphOperation import MorphOperation
from classes.MetaClasses import NonInstantiableMeta


class ContourMerge(metaclass=NonInstantiableMeta):
    def mergeContoursGenDist(contours, mergingThresh):
        mergedContours = []
        # # Merge Contours
        for contour in contours:
            x = y = w = h = 0
            try:
                x, y, w, h = cv2.boundingRect(contour)
            except:
                x, y, w, h = contour
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

    def mergeContoursTopLeftDist(contours, mergingThresh):
        mergedContours = []
        for contour in contours:
            x = y = w = h = 0
            try:
                x, y, w, h = cv2.boundingRect(contour)
            except:
                x, y, w, h = contour
            current_rect = (x, y, x + w, y + h)

            merged = False
            for i, mergedRect in enumerate(mergedContours):
                if (abs(x - mergedRect[0]) ** 2 + abs(y - mergedRect[1])) ** (
                    1 / 2
                ) < mergingThresh:
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
        return mergedContours

    def mergeContoursCenterDist(contours, mergingThresh):
        mergedContours = []

        for contour in contours:
            x = y = w = h = 0
            try:
                x, y, w, h = cv2.boundingRect(contour)
            except:
                x, y, w, h = contour

            cx = x + w // 2
            cy = y + h // 2

            current_rect = (x, y, x + w, y + h)

            merged = False
            for i, merged_rect in enumerate(mergedContours):
                # Calculate center of merged contour
                mx = (merged_rect[0] + merged_rect[2]) // 2
                my = (merged_rect[1] + merged_rect[3]) // 2

                # Calculate distance between centers
                distance = ((cx - mx) ** 2 + (cy - my) ** 2) ** 0.5

                if distance < mergingThresh:
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

"""
Helper functions for face detection
"""

from pathlib import Path
from logging import getLogger, INFO

import numpy as np
import cv2

logger = getLogger("helpers")
logger.setLevel(INFO)
logger.info("Loading face detection model.")

MODEL_FILES = Path("./model_files")
FACE_CLASSIFIER = cv2.CascadeClassifier(str(MODEL_FILES / "face_classifier.xml"))


def detect_faces(
        img: np.ndarray,
        convert_to_grayscale: bool = False,
):
    """
    Detect faces in an image.
    :param img:
    :param convert_to_grayscale:
    :return:
    """

    working_image = img.copy()
    if convert_to_grayscale:
        working_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)

    faces = FACE_CLASSIFIER.detectMultiScale(working_image, 1.3, 5)

    if not faces:
        return

    return faces


# def face_detector(img):
#     # Convert image to grayscale for faster detection
#     gray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray, 1.3, 5)
#
#     if faces is ():
#         return
#
#     allfaces = []
#     rects = []
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#         roi = img[y:y+h, x:x+w]
#         allfaces.append(roi)
#         rects.append((x,w,y,h))
#     return True, rects, allfaces, img
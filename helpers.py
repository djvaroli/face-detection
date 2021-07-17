"""
Helper functions for face detection
"""

from typing import *
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
    Returns the cropped faces, locations of the faces in the original image
    and the original image with a bounding box around the detected face.
    :param img:
    :param convert_to_grayscale:
    :return:
    """

    working_image = img.copy()
    if convert_to_grayscale:
        working_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)

    detected_faces = FACE_CLASSIFIER.detectMultiScale(working_image, 1.3, 5)

    if len(detected_faces) == 0:
        return

    just_faces = []
    for i, (x, y, w, h) in enumerate(detected_faces):
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = img[y:y + h, x: x + w]
        just_faces.append(roi)

    return just_faces, detected_faces, img


def convert_image_channels(image, to: str):
    """
    converts image color space
    :param image:
    :param to:
    :return:
    """
    op_map = {
        "rgb": cv2.COLOR_BGR2RGB,
        "bgr": cv2.COLOR_RGB2BGR
    }
    return cv2.cvtColor(image, op_map[to])


def open_image_window(
        image: Optional[np.ndarray] = None,
        path_to_image: Optional[str] = None,
        image_format: str = "bgr",
        window_name: str = "Original Image",
        wait_key_delay: int = 0
) -> None:
    """
    Opens specified image in a cv2 window
    :param image:
    :param path_to_image:
    :param window_name:
    :param image_format:
    :param wait_key_delay
    :return:
    """

    assert image is not None or path_to_image is not None, "Must specify exactly one of image or path_to_image"

    if path_to_image:
        image_bgr = cv2.imread(path_to_image)
    else:
        image_bgr = image

    if image_format != "bgr":
        image_bgr = convert_image_channels(image_bgr, to="bgr")

    # open loaded image in a separate window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image_bgr)
    cv2.waitKey(wait_key_delay)  # wait for a key to be pressed before closing window
    cv2.destroyWindow(window_name)

    return
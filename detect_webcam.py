"""
Detect face from live webcam feed
"""

from pathlib import Path

import numpy as np
import cv2

from helpers import detect_faces, open_image_window

PATH_TO_FACES = Path("./faces")


img = cv2.imread(str(PATH_TO_FACES / "obamo.jpg"))
detected_faces, face_coordinates, img = detect_faces(img)
open_image_window(img)



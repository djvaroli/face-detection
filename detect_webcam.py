"""
Detect face from live webcam feed
"""

from pathlib import Path

import numpy as np
import cv2

from helpers import detect_faces

PATH_TO_FACES = Path("./faces")


img = cv2.imread(str(PATH_TO_FACES / "obamo.jpg"))
faces = detect_faces(img)
print(faces)



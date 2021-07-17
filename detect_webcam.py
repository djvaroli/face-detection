"""
Detect face from live webcam feed
"""

from pathlib import Path

import numpy as np
import cv2

from helpers import detect_faces, open_image_window

PATH_TO_FACES = Path("./faces")


video_capture = cv2.VideoCapture(0)
cv2.namedWindow("Face Detection Live", cv2.WINDOW_NORMAL)

while True:
    _, frame = video_capture.read()
    detected_faces, face_coordinates, img = detect_faces(frame)
    cv2.imshow("Face Detection Live", img)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

video_capture.release()
cv2.destroyAllWindows()

# img = cv2.imread(str(PATH_TO_FACES / "obamo.jpg"))
# detected_faces, face_coordinates, img = detect_faces(img)
# open_image_window(img)



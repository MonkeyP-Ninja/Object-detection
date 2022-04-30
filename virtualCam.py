import pyvirtualcam
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

with pyvirtualcam.Camera(width=1280, height=720, fps=20) as cam:
    print(f'Using virtual camera: {cam.device}')
    frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB

    while True:

        success, img=cap.read()
        cam.send(img)
        cam.sleep_until_next_frame()
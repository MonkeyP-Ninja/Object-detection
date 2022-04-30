import cv2
import mediapipe as mp
import numpy as np

#setup the drawing tool
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_objectron= mp.solutions.objectron
remover = cv2.createBackgroundSubtractorKNN()


cam=cv2.VideoCapture(1)
cam.set(3,1280)
cam.set(4,720)


while True:
    ret, img = cam.read()
    img2=remover.apply(img)
    #using the mask
    rest=cv2.bitwise_and(img,img,mask=img2)
    if ret:
        cv2.imshow('Test',rest)

        if cv2.waitKey(5) & 0xFF==ord('q'):
            cam.release()
            cv2.destroyAllWindows()
            break




def backGrounRemoval():
    return None
















import cv2
import mediapipe as mp
import numpy as np

#setup the drawing tool
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_objectron= mp.solutions.objectron


cam=cv2.VideoCapture(0)
cam.set(3,1280)
cam.set(4,720)


with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as objectron:
    while True:
        ret, img=cam.read()

        if ret:
            img=cv2.flip(img,1)
            bgr=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


            results=objectron.process(bgr)
            print(results.detected_objects)
            if results.detected_objects != None:
                for objects in results.detected_objects:
                    mp_drawing.draw_landmarks(img, objects.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                    mp_drawing.draw_axis(img, objects.rotation, objects.translation)


            cv2.imshow("selfie", img)









            if cv2.waitKey(5) & 0xFF==ord('q'):
                cam.release()
                cv2.destroyAllWindows()
                break
















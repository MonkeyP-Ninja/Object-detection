import cv2
import mediapipe as mp
import numpy as np

#setup the drawing tool
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_selfie= mp.solutions.selfie_segmentation
BG_COLOR = (192, 192, 192)

cam=cv2.VideoCapture(0)
cam.set(3,1280)
cam.set(4,720)


with mp_selfie.SelfieSegmentation(model_selection=1) as selfie:
    while True:
        ret, img=cam.read()

        if ret:
            img=cv2.flip(img,1)
            bgr=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


            results=selfie.process(bgr)
            condition = np.stack(
                (results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(img.shape, dtype=np.uint8)
            bg_image[:] = cv2.blur(img,(30,30))
            output_image = np.where(condition, img, bg_image)
            cv2.imshow("selfie", output_image)

            if cv2.waitKey(5) & 0xFF==ord('q'):
                cam.release()
                cv2.destroyAllWindows()
                break

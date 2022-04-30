import cv2
import mediapipe as mp




class holistic_body:

    def __init__(self,min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic=mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)




    def draw_body(self,img):
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bodyPoints=self.holistic.process(bgr_img)
        if bodyPoints:
            self.mp_drawing.draw_landmarks(
                img,
                bodyPoints.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.
                    get_default_pose_landmarks_style())














import cv2
import mediapipe as mp


class hand_tracker:
    def __init__(self, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Build handTracker
        self.static_image_mode = static_image_mode
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.static_image_mode,
                                         min_detection_confidence=self.min_detection_confidence,
                                         min_tracking_confidence=self.min_tracking_confidence)

        # get outils for drawing
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def get_draw_landmarks(self, img):
        #mediapipe works with bgr
        bgr_img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        points_list = self.hands.process(bgr_img).multi_hand_landmarks
        print(points_list)

        if points_list:

            for points in points_list:
                self.mp_drawing.draw_landmarks(img, points, self.mp_hands.HAND_CONNECTIONS,
                                               self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                               self.mp_drawing_styles.get_default_hand_connections_style())




    #works but laggy and crashe need to optimised
    def point_by_index(self,index,img):
       if self.hands.process(img).multi_hand_landmarks:
            point=self.hands.process(img).multi_hand_landmarks[0].landmark[index]
            x, y = int(point.x*1280),int(point.y*720)
            cv2.circle(img, (x, y), 63, (0, 0, 255), -1)

            




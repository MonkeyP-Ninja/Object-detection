import cv2
import cv2 as cv2


class eyetracker :
    def __init__(self,eyes_list=[]):


        self.eyes_list=eyes_list


    def eyetrack(self,frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes =cv2.CascadeClassifier('trainedModels/haarcascade_eye.xml')
        self.eyes_list=eyes.detectMultiScale(gray_frame)

    def drawRectangles(self,frame):
       for i in self.eyes_list:
            cv2.rectangle(frame,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(255,0,0),2)















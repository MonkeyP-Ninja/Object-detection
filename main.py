
import cv2 as cv
from EyeTracker import eyetracker as e
from HandTracker import hand_tracker as hand
from body import holistic_body

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
hand=hand()
body=holistic_body()


def main():
    while( cap.isOpened() ):
        ret, frame = cap.read()
        if ret == True:
            frame = cv.flip(frame, 1)
            # eyetrack
            eye = e()
            eye.eyetrack(frame)
            eye.drawRectangles(frame)

            # get the new frame
            body.draw_body(frame)
            hand.get_draw_landmarks(frame)

            #hand.point_by_index(4,newframe)
            final = cv.hconcat([frame,frame])
            cv.imshow('frame', final)




            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()



if __name__ == '__main__':
    main()

import cv2

import cv2 as cv



cap = cv.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)



def main():
    while( cap.isOpened() ):
        ret, frame = cap.read()
        if ret == True:
            frame = cv.flip(frame, 1)

            x,y=frame.shape[0],frame.shape[1]



            cv.imshow('frame', frame)




            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()



if __name__ == '__main__':
    main()

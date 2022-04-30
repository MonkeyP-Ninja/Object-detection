import cv2
import mediapipe as mp



cam=cv2.VideoCapture(0)
faceReco=cv2.CascadeClassifier('trainedModels/face.xml')
cam.set(3,1280)
cam.set(4,720)

mp_face=mp.solutions.face_detection
mp_draw=mp.solutions.drawing_utils


with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fadec:


    while True:
        ret, i=cam.read()
        if ret:
            i=cv2.flip(i,1)
            kl=i.copy()
            y=cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
            height, width=i.shape[0],i.shape[1]
            print(height,width)
            list=fadec.process(y)
            if list != None:
                if list.detections:
                    face=list.detections[0].location_data.relative_bounding_box

                    x,y,xmin,ymin=int(face.xmin*width),int(face.ymin*height),int(face.width*width),int(face.height*height)
                    print(x,y,xmin,ymin)
                    cv2.rectangle(i,(0,0),(width,height),(0,0,0),-1)
                    cv2.rectangle(i,(x,y),(x+xmin,y+ymin),(250,250,250),-1)





                    for x in list.detections:
                        mp_draw.draw_detection(i,x)

            a = cv2.bitwise_and(kl, i)
            
            a=cv2.blur(a,(30,30))

            i2 = cv2.bitwise_not(i)
            k3 = cv2.bitwise_and(kl, i2)
            final=cv2.add(a,k3)
        
            cv2.imshow('Blur face',final)


            if cv2.waitKey(5) & 0xFF == ord('q'):
                cam.release()
                cv2.destroyAllWindows()
                break


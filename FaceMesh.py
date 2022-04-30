import cv2
import mediapipe as mp

# setup the drawing tool
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# setup my device
cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)

# Initiate holistic model

with mp_face_mesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = facemesh.process(image)

        list = results.multi_face_landmarks

        if ret == True :
            if list!= None:
                for i in list:
                    mp_drawing.draw_landmarks(img, i, mp_face_mesh.FACEMESH_TESSELATION)
                    mp_drawing.draw_landmarks(img, i, connections=mp_face_mesh.FACEMESH_CONTOURS,
                                              landmark_drawing_spec=None,
                                              connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            cv2.imshow('face_grid', img)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

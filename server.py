#Sheekar Banerjee------>> AI Engineering Lead

import socket
import time

import cv2
import mediapipe as mp
import math
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

HEADERSIZE = 100

s= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1234))
s.listen(5)

class FaceMeshDetector:
    """
    Face Mesh Detector to find 468 Landmarks using the mediapipe library.
    Helps acquire the landmark points in pixel format
    """

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        """
        :param staticMode: In static mode, detection is done on each image: slower
        :param maxFaces: Maximum number of faces to detect
        :param minDetectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode,
                                                 max_num_faces=self.maxFaces,
                                                 min_detection_confidence=self.minDetectionCon,
                                                 min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        """
        Finds face landmarks in BGR Image.
        :param img: Image to find the face landmarks in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces

    def findDistance(self,p1, p2, img=None):

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length,info, img
        else:
            return length, info


while True:
    
    #cap = cv2.VideoCapture(0)
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established!")
    
    msg = "Welcome to the Server!"
    msg = f'{len(msg):<{HEADERSIZE}}' + msg
    
    
    clientsocket.send(bytes(msg, "utf-8"))

    
        
    detector = FaceMeshDetector(maxFaces=2)
    while True:
        cap = cv2.VideoCapture(0)
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)


        if faces:
            face = faces[0]
            pointLeft = face[145]
            pointRight = face[374]
            
            w, _ = detector.findDistance(pointLeft, pointRight)
            W = 6.3

            # Finding distance
            f = 840
            d = ((W * f) / w)/100
            if d<=1.5:
                c = 3
            elif d>1.5 and d<=3:
                c = 2
            elif d>3 and d<=4.5:
                c = 1
            elif d>4.5:
                c = 0
            
            msg = f"The distance is: {d} meters and Condition is: {c}"
            msg = f'{len(msg):<{HEADERSIZE}}' + msg
            clientsocket.send(bytes(msg, "utf-8"))
            time.sleep(0.1)


            cvzone.putTextRect(img, f'Distance: {int(d)}cm',
                               (face[10][0] - 100, face[10][1] - 50),
                               scale=2)
        cv2.imshow("Vingram", img)
        cv2.waitKey(1)


        
    
    
    
    
    
    


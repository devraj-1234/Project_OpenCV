import cv2
import os
import mediapipe as mp
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import time
import HandDetectorModule as htm
import FaceDetectionModule as fdm

cap = cv2.VideoCapture(0)
hand_detector = htm.HandDetector()

while True:
    success, img = cap.read()

    if not success or img is None:
        continue

    img = cv2.flip(img, 1)
    img = hand_detector.findHands(img)
    lmList = hand_detector.findPosition(img)

    fingers = []

    if lmList:
        # Thumb
        if lmList[4][1] < lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #Rest of the fingers
        for idx in range(8, 21, 4):
            if lmList[idx][2] < lmList[idx - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)


        number = 0
        for finger in fingers:
            number += finger
        
        cv2.putText(img, f"NUMBER : {number}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Finger counting', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


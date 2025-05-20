import cv2
import mediapipe as mp
import time
import HandDetectorModule as htm

pTime = 0
cap = cv2.VideoCapture(0)
detector = htm.HandDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print("Nigga put your hand down!")
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
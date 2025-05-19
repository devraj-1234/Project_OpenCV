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
        print(lmList[4])
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"FPS : {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Sample Test", img)
    cv2.waitKey(1)
import cv2
import mediapipe as mp
import time
import FaceDetectionModule as fdm

pTime = 0
cap = cv2.VideoCapture(0)
detector = fdm.FaceDetector()

while True:
    success, img = cap.read()
    img, bboxes = detector.findFaces(img)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"FPS : {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Sample Test", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

import cv2
import os
import mediapipe as mp
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import time
import HandDetectorModule as htm
import FaceDetectionModule as fdm

listImg = os.listdir("backgrounds")
imgList = []
for imgPath in listImg:
    img = cv2.imread(f"backgrounds/{imgPath}")
    height, width = img.shape[:2]
    if (height, width) != (480, 640):
        img = cv2.resize(img, (640, 480))
    imgList.append(img)

imgIndex = 0

pTime = 0
cap = cv2.VideoCapture(0)
face_detector = fdm.FaceDetector()
hand_detector = htm.HandDetector()
segmentor = SelfiSegmentation()
x_history = []
max_history = 10

last_switch_time = 0
cooldown = 2

while True:
    success, img = cap.read()

    if not success or img is None:
        continue

    img = segmentor.removeBG(img, imgList[imgIndex % len(imgList)])

    img, bboxes = face_detector.findFaces(img)
    #print(bboxes)
    img = hand_detector.findHands(img)
    lmList = hand_detector.findPosition(img)
    if len(lmList) != 0:
        x_history.append(lmList[8][1])
        if len(x_history) > max_history:
            x_history.pop(0)
        
        if len(x_history) == max_history and time.time() - last_switch_time > cooldown:       #Swiping controls, value 'cooldown' may be changed for sensitivity
            dx = x_history[-1] - x_history[0]
            if dx > 200:
                #print("Right swipe")
                imgIndex += 1
                x_history.clear()
                last_switch_time = time.time()
            elif dx < -200:
                #print("Left swipe")
                imgIndex -= 1
                x_history.clear()
                last_switch_time = time.time()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS : {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)          #FPS Display on screen
    cv2.putText(img, f"BG : {imgIndex % len(imgList) + 1}", (540, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)       #Background index displayed on screen
    cv2.imshow("Sample Test", img)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if key == ord('a'):
        imgIndex -= 1

    if key == ord('d'):
        imgIndex += 1


cap.release()
cv2.destroyAllWindows()
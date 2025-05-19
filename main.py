
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import time

cap = cv2.VideoCapture(0)       #0 means the index of the camera devices attached to the PC, and I as of now have only my webcam
cap.set(3, 640)
cap.set(4, 480)
#cap.set(cv2.CAP_PROP_FPS, 60)

listImg = os.listdir("backgrounds")
imgList = []
for imgPath in listImg:
    img = cv2.imread(f"backgrounds/{imgPath}")
    imgList.append(img)

imgIndex = 0

segmentor = SelfiSegmentation()
pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break

    imgOut = segmentor.removeBG(img, imgList[imgIndex % len(imgList)])
    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)

    #snippet to calculate and show FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime
    cv2.putText(imgStacked, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 200), 2)


    cv2.imshow("Image", imgStacked)
    key = cv2.waitKey(1)

    if key == ord('a'):
        imgIndex -= 1
    elif key == ord('d'):
        imgIndex += 1
    elif key == ord('q'):
        break
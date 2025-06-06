import cv2
import cvzone
import FaceDetectionModule as fdm

cap = cv2.VideoCapture(0)

face_detector = fdm.FaceDetector()
filter = cv2.imread('filters/star.png', cv2.IMREAD_UNCHANGED)
filter_vertical_offset = -30

while True:
    success, img = cap.read()

    if not success:
        continue

    img, bboxes = face_detector.findFaces(img, draw=False)
    x, y, w, h = bboxes[1][0], bboxes[1][1], bboxes[1][2], bboxes[1][3]
    filter_resized = cv2.resize(filter, (w+30, h))
    img = cvzone.overlayPNG(img, filter_resized, (x-10, y + filter_vertical_offset))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
from fer import FER
import cv2

emotion_detector = FER(mtcnn = True)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    emotions = emotion_detector.detect_emotions(img_resized)
    print(emotions)

    cv2.imshow("Image with emotions ", img)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
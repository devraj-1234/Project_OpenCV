from fer import FER
import cv2

emotion_detector = FER(mtcnn=True)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    emotions = emotion_detector.detect_emotions(img)
    top_emotion, emotion_score = emotion_detector.top_emotion(img)
    print(top_emotion, emotion_score)
    cv2.imshow("Sample Test", img)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


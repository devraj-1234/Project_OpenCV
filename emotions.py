from fer import FER
import cv2
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation

segmentor = SelfiSegmentation()

emotion_detector = FER(mtcnn=True)

cap = cv2.VideoCapture(0)

emotion_colors = {
    'angry': (0, 0, 255),        # Red
    'disgust': (34, 139, 34),    # Olive Green
    'fear': (128, 0, 128),       # Dark Purple
    'happy': (0, 255, 255),      # Yellow
    'sad': (255, 0, 0),          # Blue
    'surprise': (0, 165, 255),   # Orange
    'neutral': (128, 128, 128)   # Gray
}


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    emotions = emotion_detector.detect_emotions(img)
    top_emotion, score = emotion_detector.top_emotion(img)

    if top_emotion is not None:
        bg_color = np.full((h,w,3), emotion_colors[top_emotion], dtype = np.uint8)
        output = segmentor.removeBG(img, bg_color)

    else:
        blurred = cv2.GaussianBlur(img, (23, 23), 0)
        output = segmentor.removeBG(img, blurred)

    print(top_emotion, score)
    cv2.imshow("Sample Test", output)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


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
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    emotions = emotion_detector.detect_emotions(img)
    top_emotion, score = None, 0

    if emotions:
        emotion_dict = emotions[0]["emotions"]
        top_emotion = max(emotion_dict, key=emotion_dict.get)
        score = emotion_dict[top_emotion]

    if top_emotion and score > 0.5:
        bg_color = np.full((h, w, 3), emotion_colors[top_emotion], dtype=np.uint8)
        output = segmentor.removeBG(img, bg_color)
        text = f"{top_emotion.upper()} ({score:.2f})"
    else:
        blurred = cv2.GaussianBlur(img, (23, 23), 0)
        output = segmentor.removeBG(img, blurred)
        text = "No clear emotion detected"
    cv2.putText(output, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("Emotion-Based Background", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

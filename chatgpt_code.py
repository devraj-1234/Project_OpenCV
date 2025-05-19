import cv2
import mediapipe as mp
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import cvzone

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Load background image
background = cv2.imread("1.png")
#background = cv2.resize(background, (640,480))

# Initialize segmentation
segmentor = SelfiSegmentation()

# Initialize MediaPipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip the image for natural selfie view
    img = cv2.flip(img, 1)

    # Background removal
    img_out = segmentor.removeBG(img, background)

    # Convert for mediapipe (RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Face Detection
    face_results = face_detection.process(img_rgb)
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img_out, bbox, (0, 255, 0), 2)
            cv2.putText(img_out, 'Person', (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hand Detection
    hand_results = hands.process(img_rgb)
    if hand_results.multi_hand_landmarks:
        for handLms in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img_out, handLms, mp_hands.HAND_CONNECTIONS)

    # Show Output
    cv2.imshow("Live Background Replacement", img_out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


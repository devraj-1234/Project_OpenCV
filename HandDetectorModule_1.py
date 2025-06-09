import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True, naming = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        self.handTypes = []

        if self.results.multi_hand_landmarks:
            for idx, handLms in enumerate(self.results.multi_hand_landmarks):
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

                if self.results.multi_handedness:
                    label = self.results.multi_handedness[idx].classification[0].label
                    self.handTypes.append(label)

                    h, w, _ = img.shape
                    lm = handLms.landmark[0]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if naming == True:
                        if label == "Right":
                            cv2.putText(img, "Left", (cx - 30, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        elif label == "Left":
                            cv2.putText(img, "Right", (cx - 30, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img

    def findPosition(self, img, draw=True):
        handsList = []

        if self.results.multi_hand_landmarks and self.results.multi_handedness:
            for idx, handLms in enumerate(self.results.multi_hand_landmarks):
                label = self.results.multi_handedness[idx].classification[0].label  # "Left" or "Right"
                lmList = []
                h, w, _ = img.shape
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        if id == 0:
                            cv2.circle(img, (cx, cy), 25, (255, 255, 255), cv2.FILLED)
                        elif id % 4 == 0:
                            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                handsList.append({"hand": label, "lmList": lmList})

        return handsList

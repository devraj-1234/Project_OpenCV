import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
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
        self.handTypes = []                                     

    def findHands(self, img, draw = True):
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
                    if label == "Right": 
                        cv2.putText(img, "Left", (cx - 30, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  
                    else :
                        cv2.putText(img, "Right", (cx - 30, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return img
    
    def findPosition(self, img, handNo = 0, draw = True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    if id == 0:
                        cv2.circle(img, (cx, cy), 25, (255, 255, 255), cv2.FILLED)
                    if id != 0 and id % 4 == 0:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return lmList

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, f"FPS : {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Sample Test", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()

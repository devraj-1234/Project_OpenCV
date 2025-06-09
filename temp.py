import cv2
import HandDetectorModule_1 as htm
import mediapipe as mp
import time
import cvzone

cap = cv2.VideoCapture(0)
hand_detector = htm.HandDetector()

def find_score(label, handList):
    fingers = []
    tipIds = [8, 12, 16, 20]

    def isThumbsUp(label, handList):
        # y-axis logic (higher up = lower y)
        thumb_tip_y = handList[4][2]
        thumb_ip_y = handList[3][2]

        index_tip_x = handList[8][1]
        index_mcp_x = handList[7][1]

        middle_tip_x = handList[12][1]
        middle_mcp_x = handList[11][1]

        ring_tip_x = handList[16][1]
        ring_mcp_x = handList[15][1]

        pinky_tip_x = handList[20][1]
        pinky_mcp_x = handList[19][1]

        thumb_up = thumb_tip_y < thumb_ip_y  # Upward pointing thumb

        if label == "Right":
            fingers_down = (
                index_tip_x > index_mcp_x and
                middle_tip_x > middle_mcp_x and
                ring_tip_x > ring_mcp_x and
                pinky_tip_x > pinky_mcp_x
            )

        else:
            fingers_down = (
                index_tip_x < index_mcp_x and
                middle_tip_x < middle_mcp_x and
                ring_tip_x < ring_mcp_x and
                pinky_tip_x < pinky_mcp_x
            )

        return thumb_up and fingers_down

    if isThumbsUp(label, handList):
        return [6, 0, 0, 0, 0], 6

    # For normal 5-finger scoring
    fingers.append(1 if handList[4][2] < handList[3][2] else 0)

    for id in tipIds:
        if label == "Right":
            fingers.append(1 if handList[id][1] < handList[id - 1][1] else 0)
        elif label == "Left":
            fingers.append(1 if handList[id][1] > handList[id - 1][1] else 0)

    score = sum(fingers)
    return fingers, score

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if not success or img is None:
        print("Failure")
        continue

    img = hand_detector.findHands(img, True, False)
    handsList = hand_detector.findPosition(img)

    if len(handsList) == 2:
        dict1 = handsList[0]
        dict2 = handsList[1]

        # Correctly assign based on the actual 'hand' label
        if dict1['lmList'][0][1] < dict2['lmList'][0][1]:
            leftHand = dict1['lmList']
            rightHand = dict2['lmList']

        else:
            leftHand = dict2['lmList']
            rightHand = dict1['lmList']

        # ⚠️ Match hand label to correct list
        left_score = find_score("Left", leftHand)[1]
        right_score = find_score("Right", rightHand)[1]

        # Draw scores
        cv2.putText(img, f"{left_score}", (leftHand[0][1], leftHand[0][2]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, f"{right_score}", (rightHand[0][1], rightHand[0][2]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

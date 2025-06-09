import cv2
import HandDetectorModule_1 as htm
import mediapipe as mp
import time
import cvzone

cap = cv2.VideoCapture(0)
hand_detector = htm.HandDetector()

def find_score(label, handList):            #Function to evaluate score from hand gesture, label specifies hand position
    fingers = []
    tipIds = [8, 12, 16, 20]

    def isThumbsUp(label, handList):
        #Function to check whether only the thumb is up and rest others down, to result in a six
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

        thumb_up = thumb_tip_y < thumb_ip_y  

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
    fingers.append(1 if handList[4][2] < handList[3][2] else 0)         #Thumb up or down

    for id in tipIds:                                                   #Rest four fingers logic
        if label == "Right":
            fingers.append(1 if handList[id][1] < handList[id - 1][1] else 0)       
        elif label == "Left":
            fingers.append(1 if handList[id][1] > handList[id - 1][1] else 0)

    score = sum(fingers)
    return fingers, score

scores = {'Left' : 0, 'Right' : 0}
curr_batter = 'Left'
waitingForValidInput = True
gameOver = False
last_scored_time = 0
cooldown_duration = 3

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if not success or img is None:
        print("Failure")
        continue

    img = hand_detector.findHands(img, True, False)
    handsList = hand_detector.findPosition(img)
    currentTime = time.time()

    if not gameOver and currentTime - last_scored_time > cooldown_duration:
        if len(handsList) == 2:
            dict1 = handsList[0]
            dict2 = handsList[1]

            # Assigning hands according to position relative to camera and screen
            if dict1['lmList'][0][1] < dict2['lmList'][0][1]:
                leftHand = dict1['lmList']
                rightHand = dict2['lmList']

            else:
                leftHand = dict2['lmList']
                rightHand = dict1['lmList']

            #Get scores according to the hand position
            left_score = find_score("Left", leftHand)[1]
            right_score = find_score("Right", rightHand)[1]

            if left_score != 0 and right_score != 0 :
                waitingForValidInput = False
                if left_score == right_score:
                    gameOver = True
                else:
                    scores[curr_batter] += left_score if curr_batter == 'Left' else right_score
                last_scored_time = currentTime

            else:
                waitingForValidInput = True

        else:
            waitingForValidInput = True

    if gameOver:
        cv2.putText(img, "YOU'RE OUT!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(img, "Press 'R' to Restart", (180, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(img, f"{scores[curr_batter]}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Image', img)
    
    
    key = cv2.waitKey(1)

    if key == ord('r'):
        gameOver = False
        last_scored_time = time.time()
        scores = {'Left': 0, 'Right': 0}
        waitingForValidInput = True
        curr_batter = 'Left'

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import HandDetectorModule_1 as htm
import mediapipe as mp
import time
import cvzone
import os
import numpy as np
import math



cap = cv2.VideoCapture(0)
hand_detector = htm.HandDetector()
folder_path = "gang_signs_overlays"

# Load all overlay PNGs (with resizing)
def load_overlay_images(folder_path, size=(100, 100)):
    overlays = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            key = os.path.splitext(file_name)[0]
            full_path = os.path.join(folder_path, file_name)
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                resized = cv2.resize(img, size)
                if resized.shape[2] == 4:
                    resized = resized[:, :, :3]  # Remove alpha channel if present
                overlays[key] = resized
            else:
                print(f"Warning: Could not read {full_path}")
    return overlays


def distance(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

def left_or_right(fingers):
    wrist = fingers[0][1]
    pinky = fingers[20][1]

    return "Left" if wrist < pinky else "Right"

def check_horizontal(lmList):
    x1, y1 = lmList[0][1], lmList[0][2]   # Wrist
    x2, y2 = lmList[9][1], lmList[9][2]   # Middle MCP
    dx = x2 - x1
    dy = y2 - y1
    angle_deg = math.degrees(math.atan2(dy, dx))
    return -50 < angle_deg < 50 or -179 <= angle_deg < -130 or int(angle_deg) == 180 or 130 < angle_deg <= 179
    

# Define detection logic for 'yo' sign
def yo_sign(handsList):
    fingers = handsList[0]['lmList']
    index_tip = fingers[8][2]
    middle_tip = fingers[12][2]
    ring_tip = fingers[16][2]
    pinky_tip = fingers[20][2]

    index_mcp = fingers[8-1][2]
    middle_mcp = fingers[12-2][2]
    ring_mcp = fingers[16-2][2]
    pinky_mcp = fingers[20-1][2]

    return (not check_horizontal(fingers) and index_tip < index_mcp and middle_tip > middle_mcp and ring_tip > ring_mcp and pinky_tip < pinky_mcp)


def fist_sign(handsList):
    fingers = handsList[0]['lmList']
    return (all(fingers[i][2] > fingers[i-2][2] for i in [8, 12, 16, 20]) or all(fingers[i][2] < fingers[i+1][2] for i in [5, 9, 13, 17])) and not check_horizontal(fingers)

def peace_sign(handsList):
    fingers = handsList[0]['lmList']
    return (
        not check_horizontal(fingers) and
        fingers[8][2] < fingers[6][2] and      
        fingers[12][2] < fingers[10][2] and   
        fingers[16][2] > fingers[14][2] and   
        fingers[20][2] > fingers[18][2]      
    )

def hello_sign(handsList):
    fingers = handsList[0]['lmList']
    return all(fingers[i][2] < fingers[i-2][2] for i in [4, 8, 12, 16, 20]) and not check_horizontal(fingers)


def point_up_sign(handsList):
    fingers = handsList[0]['lmList']
    return (
        not check_horizontal(fingers) and
        fingers[8][2] < fingers[6][2] and     # Index up
        fingers[12][2] > fingers[10][2] and   # Middle down
        fingers[16][2] > fingers[14][2] and   # Ring down
        fingers[20][2] > fingers[18][2]       # Pinky down
    )

def middle_finger_sign(handsList):
    fingers = handsList[0]['lmList']
    return (
        not check_horizontal(fingers) and
        fingers[8][2] > fingers[6][2] and
        fingers[12][2] < fingers[10][2] and
        fingers[16][2] > fingers[14][2] and
        fingers[20][2] > fingers[18][2]
    )

def ok_sign(handsList):
    fingers = handsList[0]['lmList']

    thumb_tip = fingers[4]
    index_tip = fingers[8]

    # Thumb and index touching
    thumb_index_touch = distance(thumb_tip, index_tip) < 25

    # Middle, ring, pinky up
    middle_up = fingers[12][2] < fingers[10][2]
    ring_up = fingers[16][2] < fingers[14][2]
    pinky_up = fingers[20][2] < fingers[18][2]

    return thumb_index_touch and middle_up and ring_up and pinky_up

def fingers_crossed_sign(handsList):
    fingers = handsList[0]['lmList']
    return  (
        not check_horizontal(fingers) and
        fingers[8][2] < fingers[7][2] and
        fingers[12][2] < fingers[11][2] and
        fingers[16][2] > fingers[14][2] and
        fingers[20][2] > fingers[18][2] and
        abs(fingers[8][1] - fingers[12][1]) < 5
    )

def thumbs_up_sign(handsList):
    fingers = handsList[0]['lmList']
    horizontal = check_horizontal(fingers)
    ans = False
    if horizontal:
        side = left_or_right(fingers)
        if side == 'Left':
            ans = (
                fingers[4][2] < fingers[3][2] and
                fingers[8][1] < fingers[6][1] and
                fingers[12][1] < fingers[10][1] and
                fingers[16][1] < fingers[14][1] and
                fingers[20][1] < fingers[18][1]
            )

        else:
            ans = (
                fingers[4][2] < fingers[3][2] and
                fingers[8][1] > fingers[6][1] and
                fingers[12][1] > fingers[10][1] and
                fingers[16][1] > fingers[14][1] and
                fingers[20][1] > fingers[18][1]
            )

    return ans

def gun_sign(handsList):
    fingers = handsList[0]['lmList']
    horizontal = check_horizontal(fingers)
    ans = False
    if horizontal:
        side = left_or_right(fingers)
        if side == 'Left':
            ans = (
                fingers[8][1] > fingers[6][1] and
                fingers[12][1] > fingers[10][1] and
                fingers[16][1] < fingers[14][1] and
                fingers[20][1] < fingers[18][1]
            )

        else:
            ans = (
                fingers[8][1] < fingers[6][1] and
                fingers[12][1] < fingers[10][1] and
                fingers[16][1] > fingers[14][1] and
                fingers[20][1] > fingers[18][1]
            )

    return ans

def call_sign(handsList):
    fingers = handsList[0]['lmList']
    horizontal = check_horizontal(fingers)
    ans = False
    if horizontal:
        side = left_or_right(fingers)
        if side == 'Left':
            ans = (
                fingers[4][2] < fingers[2][2] and
                fingers[8][1] < fingers[6][1] and
                fingers[12][1] < fingers[10][1] and
                fingers[16][1] < fingers[14][1] and
                fingers[20][1] > fingers[19][1]
            )

        else:
            ans = (
                fingers[4][2] < fingers[2][2] and
                fingers[8][1] > fingers[6][1] and
                fingers[12][1] > fingers[10][1] and
                fingers[16][1] > fingers[14][1] and
                fingers[20][1] < fingers[19][1]
            )

    return ans

def thumbs_down_sign(handsList):
    fingers = handsList[0]['lmList']
    horizontal = check_horizontal(fingers)
    ans = False
    if horizontal:
        side = left_or_right(fingers)
        if side == 'Left':
            ans = (
                fingers[4][2] > fingers[3][2] and
                fingers[8][1] < fingers[6][1] and
                fingers[12][1] < fingers[10][1] and
                fingers[16][1] < fingers[14][1] and
                fingers[20][1] < fingers[18][1]
            )

        else:
            ans = (
                fingers[4][2] > fingers[3][2] and
                fingers[8][1] > fingers[6][1] and
                fingers[12][1] > fingers[10][1] and
                fingers[16][1] > fingers[14][1] and
                fingers[20][1] > fingers[18][1]
            )

    return ans



overlays = load_overlay_images(folder_path)

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failure")
        continue

    img = cv2.flip(img, 1)
    img = hand_detector.findHands(img, True, False)
    handsList = hand_detector.findPosition(img)

    if handsList:
        if yo_sign(handsList):
            img[0:100, 0:100] = overlays['yo_sign']

        elif thumbs_up_sign(handsList):
            img[0:100, 0:100] = overlays['thumbs_up_sign']

        elif fist_sign(handsList):
            img[0:100, 0:100] = overlays['fist_sign']

        elif fingers_crossed_sign(handsList):
            img[0:100, 0:100] = overlays['fingers_crossed_sign']

        elif peace_sign(handsList):
            img[0:100, 0:100] = overlays['peace_sign']

        elif hello_sign(handsList):
            img[0:100, 0:100] = overlays['hello_sign']

        elif point_up_sign(handsList):
            img[0:100, 0:100] = overlays['point_up_sign']

        elif middle_finger_sign(handsList):
            img[0:100, 0:100] = overlays['middle_finger_sign']

        elif ok_sign(handsList):
            img[0:100, 0:100] = overlays['ok_sign']

        elif gun_sign(handsList):
            img[0:100, 0:100] = overlays['gun_sign']

        elif call_sign(handsList):
            img[0:100, 0:100] = overlays['call_sign']

        elif thumbs_down_sign(handsList):
            img[0:100, 0:100] = overlays['thumbs_down_sign']
            
    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

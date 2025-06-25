import cv2
import HandDetectorModule_1 as htm
import mediapipe as mp
import time
import cvzone
import os
import numpy as np
import math
import pyperclip
import pyautogui as pg



cap = cv2.VideoCapture(0)                                                                    
hand_detector = htm.HandDetector()
folder_path = "gang_signs_overlays"
last_trigger_time = 0
cooldown = 3
typing_state = False

def type_emoji(emoji):
    pyperclip.copy(emoji)
    pg.hotkey("ctrl", "v")
    #pg.press("enter")

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
        abs(fingers[8][1] - fingers[12][1]) < 10
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

emoji_yo_sign = "🤘"              # Rock on / Yo!
emoji_fist_sign = "✊"            # Fist
emoji_peace_sign = "✌️"           # Peace
emoji_hello_sign = "👋"           # Open palm
emoji_point_up_sign = "👆"        # Pointing up
emoji_middle_finger_sign = "🖕"   # Middle finger (use responsibly)
emoji_ok_sign = "👌"              # OK gesture
emoji_fingers_crossed_sign = "🤞" # Fingers crossed
emoji_thumbs_up_sign = "👍"       # Thumbs up
emoji_gun_sign = "🔫"             # Gun (or ✌️ sideways in context)
emoji_call_sign = "🤙"            # Shaka / Call me gesture
emoji_thumbs_down_sign = "👎"     # Thumbs down



overlays = load_overlay_images(folder_path)
curr_overlay = None

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failure")
        continue

    img = cv2.flip(img, 1)
    img = hand_detector.findHands(img, True, False)
    handsList = hand_detector.findPosition(img)
    current_time = time.time()

    if typing_state:

        cv2.putText(img, "Typing active! Press 't' to toggle", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if handsList:

            if yo_sign(handsList) and current_time - last_trigger_time > cooldown:
                type_emoji(emoji_yo_sign)
                curr_overlay = overlays['yo_sign']
                last_trigger_time = current_time

            elif thumbs_up_sign(handsList) and current_time - last_trigger_time > cooldown:
                type_emoji(emoji_thumbs_up_sign)
                curr_overlay = overlays['thumbs_up_sign']
                last_trigger_time = current_time

            elif fist_sign(handsList) and current_time - last_trigger_time > cooldown:
                type_emoji(emoji_fist_sign)
                curr_overlay = overlays['fist_sign']
                last_trigger_time = current_time

            elif fingers_crossed_sign(handsList) and current_time - last_trigger_time > cooldown:
                type_emoji(emoji_fingers_crossed_sign)
                curr_overlay = overlays['fingers_crossed_sign']
                last_trigger_time = current_time

            elif peace_sign(handsList) and current_time - last_trigger_time > cooldown:
                type_emoji(emoji_peace_sign)
                curr_overlay = overlays['peace_sign']
                last_trigger_time = current_time

            elif hello_sign(handsList) and current_time - last_trigger_time > cooldown:
                type_emoji(emoji_hello_sign)
                curr_overlay = overlays['hello_sign']
                last_trigger_time = current_time

            elif point_up_sign(handsList) and current_time - last_trigger_time > cooldown:
                type_emoji(emoji_point_up_sign)
                curr_overlay = overlays['point_up_sign']
                last_trigger_time = current_time

            elif middle_finger_sign(handsList) and current_time - last_trigger_time > cooldown:
                type_emoji(emoji_middle_finger_sign)
                curr_overlay = overlays['middle_finger_sign']
                last_trigger_time = current_time

            elif ok_sign(handsList) and current_time - last_trigger_time > cooldown:
                type_emoji(emoji_ok_sign)
                curr_overlay = overlays['ok_sign']
                last_trigger_time = current_time

            elif gun_sign(handsList) and current_time - last_trigger_time > cooldown:
                type_emoji(emoji_gun_sign)
                curr_overlay = overlays['gun_sign']
                last_trigger_time = current_time

            elif call_sign(handsList) and current_time - last_trigger_time > cooldown:
                type_emoji(emoji_call_sign)
                curr_overlay = overlays['call_sign']
                last_trigger_time = current_time

            elif thumbs_down_sign(handsList) and current_time - last_trigger_time > cooldown:
                type_emoji(emoji_thumbs_down_sign)
                curr_overlay = overlays['thumbs_down_sign']
                last_trigger_time = current_time


    else:
        cv2.putText(img, "Typing inactive. Press 't' to toggle", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if curr_overlay is not None:
        img[0:100, 0:100] = curr_overlay

    # Draw mode indicator rectangle (top-right corner)
    color = (0, 255, 0) if typing_state else (0, 0, 255)
    cv2.rectangle(img, (540, 10), (630, 50), color, -1)
    cv2.putText(img, "TYP", (545, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    if typing_state :
        cooldown_over = (current_time - last_trigger_time) > cooldown
        cool_color = (255, 0, 0) if cooldown_over else (0, 255, 255)
        cv2.rectangle(img, (460, 10), (530, 50), cool_color, -1)
        cv2.putText(img, "CD", (470, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow('Image', img)

    key = cv2.waitKey(1)

    if key == ord('t'):
        typing_state = not typing_state

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

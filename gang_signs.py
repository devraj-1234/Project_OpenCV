import cv2
import HandDetectorModule_1 as htm
import mediapipe as mp
import time
import cvzone
import os
import numpy as np
import math

def distance(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

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

    return (index_tip < index_mcp and middle_tip > middle_mcp and ring_tip > ring_mcp and pinky_tip < pinky_mcp)


def fist_sign(handsList):
    fingers = handsList[0]['lmList']
    return all(fingers[i][2] > fingers[i-2][2] for i in [8, 12, 16, 20]) or all(fingers[i][2] < fingers[i+1][2] for i in [5, 9, 13, 17])

def peace_sign(handsList):
    fingers = handsList[0]['lmList']
    return (
        fingers[8][2] < fingers[6][2] and      
        fingers[12][2] < fingers[10][2] and   
        fingers[16][2] > fingers[14][2] and   
        fingers[20][2] > fingers[18][2]      
    )

def hello_sign(handsList):
    fingers = handsList[0]['lmList']
    return all(fingers[i][2] < fingers[i-2][2] for i in [4, 8, 12, 16, 20])


def point_up_sign(handsList):
    fingers = handsList[0]['lmList']
    return (
        fingers[8][2] < fingers[6][2] and     # Index up
        fingers[12][2] > fingers[10][2] and   # Middle down
        fingers[16][2] > fingers[14][2] and   # Ring down
        fingers[20][2] > fingers[18][2]       # Pinky down
    )


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

        elif fist_sign(handsList):
            img[0:100, 0:100] = overlays['fist_sign']

        elif peace_sign(handsList):
            img[0:100, 0:100] = overlays['peace_sign']

        elif hello_sign(handsList):
            img[0:100, 0:100] = overlays['hello_sign']

        elif point_up_sign(handsList):
            img[0:100, 0:100] = overlays['point_up_sign']


            
    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

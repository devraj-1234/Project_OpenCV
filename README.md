# 🧠 Computer Vision R&D Project 2025

This repository contains four real-time Computer Vision applications built using **Python**, **OpenCV**, and **MediaPipe**, aimed at demonstrating gesture and face-based interaction systems. These mini-projects cover a wide range of use-cases from gaming to emotion detection and interactive background manipulation.

## 📁 Project Structure

- `with_bg_and_swipe.py` – Real-time **Background Removal** with hand swipe gestures to change background.
- `emotions.py` – **Facial Emotion Recognition (FER)** using deep learning for real-time emotion classification.
- `hand_cricket.py` – A two-player **Hand Cricket** game using gesture-based interaction.
- `gang_sign.py` – **Gang Sign Typing** system that detects hand gestures and types corresponding words or overlays PNGs.

---

## 🔧 Requirements and Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/devraj-1234/Computer-Vision.git
   cd your-repo
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   **On Windows**
   ```bash
   .venv\Scripts\activate
   ```

   **On MacOS/Linux**
   ```bash
   source venv/bin/activate
   ```


4. **Install all required dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

```bash
python with_bg_and_swipe.py      # For background remover
python emotions.py               # For emotion recognition
python hand_cricket.py           # For gesture-based hand cricket
python gang_sign.py              # For gang sign typing system
```
---

## 🔧 Major Libraries Used

- `opencv-python` – For real-time video capture and image processing
- `mediapipe` – For hand, face, and body landmark detection
- `numpy` – For numerical operations
- `tensorflow` / `keras` – For facial emotion classification model
- `cvzone` – Optional utilities for drawing and overlays
- `os`, `time`, `random` – Standard libraries for logic control and file management

---

## 🎮 Features

### 1. Background Remover + Hand Swipe (`with_bg_and_swipe.py`)
- Removes the background using segmentation.
- Allows users to swipe left/right (using hand gestures) to change the background image.
- All backgrounds are stored in a list variable and can be customized.
- More backgrounds can be added in the `backgrounds\` folder.

### 2. Facial Emotion Recognition (`emotions.py`)
- Detects facial landmarks and classifies emotions into categories like **Happy**, **Sad**, **Angry**, etc.
- Displays emoji or text feedback on the screen.
- Useful for emotion-based UI/UX systems.

### 3. Hand Cricket Game (`hand_cricket.py`)
- Two-player game inspired by childhood hand cricket.
- Toss logic included (manual/random).
- Uses gesture input for runs and choices (bat/ball).
- Handles innings, scoring, and winner declaration.

### 4. Gang Sign Typing (`gang_sign.py`)
- Detects custom hand signs. 
- Preset gang signs are : 
  - Yo 🤘
  - Fist ✊
  - Peace ✌️
  - Hello/Waving 👋
  - Point up 👆
  - Ok 👌
  - Fingers crossed 🤞
  - Thumbs up 👍
  - Gun 🔫
  - Call/Shaka 🤙
  - Thumbs down 👎
- Displays corresponding PNG overlays in the top-left corner and types the respective emoji into the active textbox.
- Cooldown mechanism avoids repeated detection spam.


---


## Hope you enjoy running and playing around with this project! 👾
Feel free to contribute or suggest new features. 👍
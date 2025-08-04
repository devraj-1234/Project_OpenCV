<div id="top"></div>

<!-- PROJECT SHIELDS -->
<div align="center">

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

</div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/devraj-1234/Computer_Vision">
    <img width="140" alt="project logo" src="/.github/logo.png">
  </a>

  <h3 align="center">Computer Vision R&D Project 2025</h3>

  <p align="center">
    <i>Gesture and Emotion-based Real-Time Interaction Systems</i>
  </p>
</div>

---

<details>
<summary>📑 Table of Contents</summary>

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)
  - [Maintainer(s)](#maintainers)
  - [Creator(s)](#creators)
- [Additional Documentation](#additional-documentation)

</details>

---

## 📌 About The Project

<div align="center">
  <img width="80%" alt="screenshot" src="https://via.placeholder.com/800x300.png?text=Computer+Vision+Showcase">
</div>

This repository contains four real-time Computer Vision mini-projects built using **Python**, **OpenCV**, and **MediaPipe**. These projects demonstrate gesture and face-based interaction systems for applications such as background manipulation, emotion detection, hand-gesture games, and gesture-based typing.

### 🧠 Projects Overview

- 🎭 **Background Remover with Hand Swipe** – Switch backgrounds via hand gestures.
- 😊 **Facial Emotion Recognition** – Detect emotions and display emojis.
- 🏏 **Hand Cricket Game** – Play cricket using hand gestures.
- 🤟 **Gang Sign Typing System** – Detect signs and overlay PNGs or type emojis.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## 🚀 Getting Started

To set up a local instance of the application, follow these steps.

### 🔧 Prerequisites

- Python 3.8+
- pip (Python package installer)

### 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/devraj-1234/Computer_Vision.git
   cd Computer_Vision
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   **On Windows:**

   ```bash
   .\venv\Scripts\activate
   ```

   **On MacOS/Linux:**

   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

---

## 🎮 Usage

### Run individual projects:

```bash
python with_bg_and_swipe.py      # Background changer
python emotions.py               # Facial emotion recognition
python hand_cricket.py           # Gesture-based hand cricket
python gang_sign.py              # Gang sign to emoji/PNG typer
```

### 🛠 Major Libraries Used

- `opencv-python` – Image processing
- `mediapipe` – Landmark detection
- `tensorflow`, `keras` – Emotion classification
- `numpy`, `os`, `random`, `time` – Utilities
- `cvzone` – Drawing overlays and helpers

### 🧩 Features Summary

#### 1. Background Remover + Hand Swipe

- Segment person from background
- Change background via left/right swipe gestures
- Customizable `backgrounds/` folder

#### 2. Facial Emotion Recognition

- Detects face and classifies emotions: Happy, Sad, Angry, etc.
- Displays emojis or text overlay

#### 3. Hand Cricket Game

- 2-player game using hand gestures
- Toss system (manual/random)
- Tracks innings, scores, winner

#### 4. Gang Sign Typing System

- Detects signs like: ✌️ 🤘 👋 👌 👍 ✊ 🤙 🔫 etc.
- Types emoji or overlays PNG in real time
- Cooldown to prevent repeated input

<p align="right">(<a href="#top">back to top</a>)</p>

---

## 👨‍💻 Contact

### Maintainer(s)

- [Dev Raj](https://github.com/devraj-1234)

### Creator(s)

- [Dev Raj](https://github.com/devraj-1234)

<p align="right">(<a href="#top">back to top</a>)</p>

---

## 📄 Additional Documentation

- [License](/LICENSE)
- [Code of Conduct](/.github/CODE_OF_CONDUCT.md)
- [Security Policy](/.github/SECURITY.md)
- [Contribution Guidelines](/.github/CONTRIBUTING.md)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->

[contributors-shield]: https://img.shields.io/github/contributors/devraj-1234/Computer_Vision.svg?style=for-the-badge
[contributors-url]: https://github.com/devraj-1234/Computer_Vision/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/devraj-1234/Computer_Vision.svg?style=for-the-badge
[forks-url]: https://github.com/devraj-1234/Computer_Vision/network/members
[stars-shield]: https://img.shields.io/github/stars/devraj-1234/Computer_Vision.svg?style=for-the-badge
[stars-url]: https://github.com/devraj-1234/Computer_Vision/stargazers
[issues-shield]: https://img.shields.io/github/issues/devraj-1234/Computer_Vision.svg?style=for-the-badge
[issues-url]: https://github.com/devraj-1234/Computer_Vision/issues
[license-shield]: https://img.shields.io/github/license/devraj-1234/Computer_Vision.svg?style=for-the-badge
[license-url]: https://github.com/devraj-1234/Computer_Vision/blob/main/LICENSE

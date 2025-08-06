# Real-Time Gesture Control & Recognition System

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img alt="MediaPipe" src="https://img.shields.io/badge/MediaPipe-007F73?style=for-the-badge&logo=google&logoColor=white"/>
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  This repository contains a dual-application project for real-time hand gesture recognition using MediaPipe. It includes a Python desktop application for controlling your computer's cursor via gestures, and a modern Streamlit web app for live gesture classification and visualization.
</p>

---

## 🚀 Project Showcase

This project offers two distinct applications built from the same core gesture recognition engine.

| 🖥️ Desktop Gesture Controller                                                                                                   | 🌐 Streamlit Recognition Web App                                                                                               |
| --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| A native desktop application that translates hand gestures into real-time mouse movements, clicks, and scrolling.               | A modern, browser-based dashboard that provides a live webcam feed with gesture classification, metrics, and an event log.   |
| <img src="https://github.com/user-attachments/assets/c529d20c-03d3-46bd-85d0-37ba7ffefb73" alt="Desktop Controller Screenshot"/> | <img src="https://github.com/user-attachments/assets/74b94fcf-b676-4743-9891-b040085a5874" alt="Streamlit App Screenshot"/> |

---

## ✨ Features

### Desktop Gesture Controller (`desktop_controller.py`)
-   **Cursor Control:** Move your mouse pointer by moving your index finger.
-   **Click Simulation:** Perform a left-click by bringing your thumb and index finger together.
-   **Scroll Functionality:** Use a designated gesture to scroll up and down.
-   **Customizable Actions:** Easily map different gestures to various keyboard or mouse actions via `PyAutoGUI`.

### Streamlit Web App (`app_streamlit.py`)
-   **Live Webcam Feed:** Uses `streamlit-webrtc` to provide a real-time video stream directly in the browser.
-   **Real-Time Classification:** Displays the detected gesture and confidence score as they happen.
-   **Performance Metrics:** Shows live FPS and processing latency.
-   **Gesture Log:** Records a timestamped log of all detected gestures for review.
-   **Adjustable Settings:** Allows users to change the camera resolution and confidence thresholds from the UI.

---

## 🛠️ Technology Stack

-   **Core Engine:** **Python**, **OpenCV**, **MediaPipe**
-   **Desktop Application:** **PyAutoGUI**
-   **Web Application:** **Streamlit**, **streamlit-webrtc**, **Pillow**, **AV**

---

## 🚀 Getting Started

Follow these steps to set up and run both applications locally.

### Prerequisites
-   Python 3.8 or higher
-   Git
-   A connected webcam

### Installation Guide

1.  **Clone the Repository**
    ```sh
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Rename Project Files** (Recommended for clarity)
    ```sh
    # On macOS/Linux
    mv latest.py desktop_controller.py
    mv streamlit_only_gusture.py app_streamlit.py

    # On Windows
    ren latest.py desktop_controller.py
    ren streamlit_only_gusture.py app_streamlit.py
    ```

3.  **Set Up a Virtual Environment**
    ```sh
    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # On Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

4.  **Download the MediaPipe Gesture Model**
    -   This project requires the `gesture_recognizer.task` model file.
    -   Download it directly from Google's official source: **[gesture_recognizer.task](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task)**
    -   Place the downloaded file in the **root directory** of your project.

5.  **Install All Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

---

## 🔧 Usage Guide

### To run the Desktop Gesture Controller:
Ensure no other application is using your webcam, then run the following command in your terminal:
```sh
python desktop_controller.py```
Press `q` on the OpenCV window to quit.

### To launch the Streamlit Web App:
Run the following command in your terminal:
```sh
streamlit run app_streamlit.py
```
This will open a new tab in your web browser with the application running.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

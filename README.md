# Sign Language & Hand Gesture Recognition

This repository contains two main applications for real-time hand gesture and sign language recognition using your webcam:

- **latest.py**: A desktop application using OpenCV and MediaPipe for gesture-based control (e.g., cursor, clicks, scroll).
- **streamlit_only_gusture.py**: A modern Streamlit web app for live gesture recognition with a professional UI.

---

## Features

- Real-time hand gesture detection via webcam
- Uses [MediaPipe](https://mediapipe.dev/) for gesture recognition
- Customizable gesture mapping and sensitivity
- Streamlit app with live video, metrics, and gesture logs
- Adjustable camera resolution for better tracking

---

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- PyAutoGUI (for desktop control)
- Streamlit (for web app)
- streamlit-webrtc (for webcam in Streamlit)
- av, pillow, numpy

Install all dependencies:
```
pip install -r requirements.txt
```

---

## Usage

### Desktop Gesture Control

```
python latest.py
```
- Press `q` to quit.
- Use hand gestures to control your mouse and perform actions.

### Streamlit Web App

```
streamlit run streamlit_only_gusture.py
```
- Open the provided local URL in your browser.
- See live gesture detection and metrics.

---

## Results:

![Screenshot (56)](https://github.com/user-attachments/assets/2e7d410f-1c0a-4102-aa85-c3c65db177ba)

![Screenshot (57)](https://github.com/user-attachments/assets/f320469c-a47c-46e0-8219-4c513268f584)

## Notes

- Make sure your webcam is connected.
- Adjust camera resolution or gesture settings in the code if needed.
- The MediaPipe model file (`gesture_recognizer.task`) must be present in the project directory.

---

## License

MIT

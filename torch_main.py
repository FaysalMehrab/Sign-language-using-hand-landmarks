import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os  # Import the 'os' module

# Define your model architecture to match the input and output sizes
class CustomNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomNN, self).__init__()
        self.fc1 = nn.Linear(64*64*3, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set the path to your 'data' folder
data_folder = 'data'  # Adjust this path based on your directory structure

# Define image transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Open the camera
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2)

# Load your PyTorch model
num_classes = len(os.listdir(data_folder))  # Update based on the number of classes
model = CustomNN(num_classes)
model.load_state_dict(torch.load('torch_model.pth'))
model.eval()  # Set the model to evaluation mode

labels_dict = {0: 'Free', 1: 'Option', 2: 'Back', 3: 'Open'}  # Adjust class labels as needed

while True:
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to read a frame from the camera.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Preprocess the data for the PyTorch model
        transformed_data = transform(frame)  # Apply the same transform as used in training
        data_aux_tensor = transformed_data.view(1, -1, 64*64*3)

        with torch.no_grad():
            prediction_proba = model(data_aux_tensor)
        prediction_label_index = torch.argmax(prediction_proba, dim=1).item()
        prediction_confidence = prediction_proba[0, prediction_label_index].item()

        predicted_character = labels_dict[prediction_label_index]

        # if predicted_character == 'Open':
        #     pyautogui.doubleClick()  # Double click
        #     time.sleep(0.19)
        # elif predicted_character == 'Option':
        #     pyautogui.rightClick()  # Right click
        #     time.sleep(0.19)
        # elif predicted_character == 'Back':
        #     pyautogui.press('space')  # Move the cursor to the center of the screen
        #     time.sleep(0.19)
        # elif predicted_character == 'Free':
        #     screen_width = pyautogui.size().width
        #     pyautogui.moveTo(screen_width - x1, y1, duration=0.005)
        #     time.sleep(0.19)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        # Set the color based on the confidence level.
        if prediction_confidence > 0.9:
            color = (0, 255, 0)  # Green for confidence > 90%
        elif prediction_confidence > 0.75:
            color = (0, 255, 255)  # Yellow for confidence between 75% and 89%
        else:
            color = (0, 0, 255)  # Red for confidence < 75%

        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3,
                    cv2.LINE_AA)

        # Display the confidence at the top left corner of the frame.
        cv2.putText(frame, f'Confidence: {prediction_confidence * 100:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

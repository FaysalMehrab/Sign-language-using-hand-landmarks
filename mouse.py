import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2)

labels_dict = {0: 'Open', 1: 'Option',2:'Free' ,3: 'Back'}

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

        prediction_proba = model.predict_proba([np.asarray(data_aux)])
        prediction_confidence = np.max(prediction_proba)
        prediction_label_index = np.argmax(prediction_proba)
        
        predicted_character = labels_dict[prediction_label_index]

        if predicted_character == 'Open':
            pyautogui.doubleClick()  # Double click
            time.sleep(0.19) 
        elif predicted_character == 'Option':
            pyautogui.rightClick()  # Right click
            time.sleep(0.19) 
        elif predicted_character == 'Back':
    # Move the cursor to the center of the screen
             pyautogui.press('space')
             time.sleep(0.19) 
             
  # Move 100 pixels to the left
        elif predicted_character == 'Free':
            # Move the cursor to the position of the hand
            #pyautogui.moveTo(x1, y1)
            time.sleep(0.19) 
            screen_width = pyautogui.size().width
            pyautogui.moveTo(screen_width - x1, y1)
            pyautogui.moveTo(screen_width - x1, y1, duration=0.005)



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

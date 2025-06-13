import cv2
import numpy as np
import pyautogui
import time
import argparse
import queue
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

class GestureControl:
    def __init__(self):
        self.args = self.parse_arguments()
        self.setup_gesture_mapping()
        self.initialize_mediapipe()
        self.setup_camera()
        self.setup_display()
        self.initialize_state()

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--headless', action='store_true', help='Run without display')
        parser.add_argument('--sensitivity', type=float, default=0.7, 
                          help='Gesture confidence threshold (0.5-1.0)')
        parser.add_argument('--control_speed', type=float, default=2.0,
                          help='Cursor control speed multiplier')
        return parser.parse_args()

    def setup_gesture_mapping(self):
        self.GESTURE_MAPPING = {
            'Thumb_Up': 'space',
            'Victory': 'right_click',
            'Open_Palm': 'left_click',
            'Pointing_Up': 'cursor_control',
            'Closed_Fist': 'drag',
            'ILoveYou': 'scroll'
        }
        self.SMOOTHING_FACTOR = 0.7
        self.PINCH_THRESHOLD = 0.05

    def initialize_mediapipe(self):
        model_path = 'gesture_recognizer.task'
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=self.process_result
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.result_queue = queue.Queue()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

    def process_result(self, result, output_image, timestamp_ms):
        self.result_queue.put((result, output_image, timestamp_ms))

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # Try Full HD first
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # Fallback to HD if needed
        if actual_width < 1920 or actual_height < 1080:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open camera.")
        print(f"Camera resolution set to: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    def setup_display(self):
        self.display_enabled = not getattr(self, 'args', None) or not getattr(self.args, 'headless', False)
        if self.display_enabled:
            try:
                cv2.namedWindow('Gesture Control', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Gesture Control', 1280, 720)
            except:
                self.display_enabled = False
                print("Display not supported. Running headless.")

    def initialize_state(self):
        screen_w, screen_h = pyautogui.size()
        self.prev_x, self.prev_y = screen_w // 2, screen_h // 2
        self.scroll_active = False
        self.drag_active = False
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=2)
        self.current_frame = None

    def capture_frames(self):
        timestamp = 0
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue
            if self.frame_queue.full():
                self.frame_queue.get()
            self.frame_queue.put((frame, timestamp))
            timestamp += 1
            time.sleep(0.01)

    def convert_to_normalized_landmarks(self, hand_landmarks):
        """Convert MediaPipe Tasks format to standard MediaPipe format"""
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        for landmark in hand_landmarks:
            new_landmark = landmark_list.landmark.add()
            new_landmark.x = landmark.x
            new_landmark.y = landmark.y
            new_landmark.z = landmark.z
        return landmark_list

    def draw_landmarks_and_info(self, frame, result):
        if not result.hand_landmarks:
            return frame

        annotated_frame = frame.copy()
        
        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            # Convert landmarks to the format expected by drawing utilities
            landmark_list = self.convert_to_normalized_landmarks(hand_landmarks)
            
            # Draw hand landmarks
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                landmark_list,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))
            
            # Draw bounding box
            x_coords = [landmark.x for landmark in hand_landmarks]
            y_coords = [landmark.y for landmark in hand_landmarks]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            frame_height, frame_width = annotated_frame.shape[:2]
            x1, y1 = int(min_x * frame_width), int(min_y * frame_height)
            x2, y2 = int(max_x * frame_width), int(max_y * frame_height)
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw gesture information
            if idx < len(result.gestures):
                gesture = result.gestures[idx][0]
                cv2.putText(annotated_frame, f"{gesture.category_name} ({gesture.score:.2f})",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                           (0, 255, 0), 2, cv2.LINE_AA)

        return annotated_frame

    def process_gestures(self, result, frame):
        if not result.gestures:
            return frame

        annotated_frame = self.draw_landmarks_and_info(frame, result)

        for hand_idx, gesture_list in enumerate(result.gestures):
            top_gesture = gesture_list[0]
            if top_gesture.score < self.args.sensitivity:
                continue

            gesture_name = top_gesture.category_name
            hand_landmarks = result.hand_landmarks[hand_idx]

            if gesture_name in self.GESTURE_MAPPING:
                action = self.GESTURE_MAPPING[gesture_name]
                self.execute_action(action, hand_landmarks)

        return annotated_frame

    def execute_action(self, action, landmarks):
        if action == 'cursor_control' and landmarks:
            self.control_cursor(landmarks)
        elif action == 'left_click':
            pyautogui.click()
        elif action == 'right_click':
            pyautogui.rightClick()
        elif action == 'drag' and not self.drag_active:
            pyautogui.mouseDown()
            self.drag_active = True
        elif action == 'scroll':
            self.scroll_active = not self.scroll_active
        elif action == 'space':
            pyautogui.press('space')

    def control_cursor(self, landmarks):
        index_tip = landmarks[8]
        thumb_tip = landmarks[4]
        
        screen_w, screen_h = pyautogui.size()
        cursor_x = int(index_tip.x * screen_w * self.args.control_speed)
        cursor_y = int(index_tip.y * screen_h * self.args.control_speed)
        
        smoothed_x = self.prev_x * self.SMOOTHING_FACTOR + cursor_x * (1 - self.SMOOTHING_FACTOR)
        smoothed_y = self.prev_y * self.SMOOTHING_FACTOR + cursor_y * (1 - self.SMOOTHING_FACTOR)
        
        pyautogui.moveTo(smoothed_x, smoothed_y)
        self.prev_x, self.prev_y = smoothed_x, smoothed_y
        
        pinch_distance = abs(index_tip.x - thumb_tip.x) + abs(index_tip.y - thumb_tip.y)
        if pinch_distance < self.PINCH_THRESHOLD:
            pyautogui.click()

    def run(self):
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        capture_thread.start()

        try:
            while True:
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue
                    
                frame, timestamp = self.frame_queue.get()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create MediaPipe Image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Process recognition
                self.recognizer.recognize_async(mp_image, timestamp)
                
                # Check for results and process frame
                annotated_frame = frame.copy()
                if not self.result_queue.empty():
                    result, output_image, _ = self.result_queue.get()
                    annotated_frame = self.process_gestures(result, frame)

                if self.display_enabled:
                    cv2.imshow('Gesture Control', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            print("\nExiting gracefully...")

        finally:
            self.stop_event.set()
            capture_thread.join(timeout=1.0)
            self.cap.release()
            self.recognizer.close()
            if self.display_enabled:
                cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = GestureControl()
    controller.run()
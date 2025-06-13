import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tempfile
import time
from PIL import Image
from mediapipe.framework.formats import landmark_pb2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# Custom CSS for professional styling
st.markdown("""
<style>
    :root {
        --primary: #4f8bf9;
        --secondary: #6c757d;
        --success: #28a745;
        --info: #17a2b8;
        --warning: #ffc107;
        --danger: #dc3545;
        --light: #f8f9fa;
        --dark: #343a40;
    }
    
    .stApp {
        max-width: 1200px;
        padding: 2rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .header {
        color: var(--primary);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
        font-size: 2.5rem;
    }
    
    .sidebar .sidebar-content {
        background-color: var(--light);
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
    }
    
    .metric-card {
        background-color: black;
        border-radius: 0.5rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
        border-left: 4px solid var(--primary);
    }
    
    .gesture-badge {
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 2rem;
        background-color: var(--light);
        display: inline-block;
        font-size: 0.9rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
    }
    
    .highlight-badge {
        background-color: var(--primary);
        color: white;
    }
    
    .video-container {
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15);
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Enhanced gesture mapping with confidence thresholds
GESTURE_CONFIG = {
    'Open_Palm': {
        "display": "🖐️ Open Palm",
        "min_confidence": 0.65,
        "color": (46, 204, 113)  # Green
    },
    'Closed_Fist': {
        "display": "✊ Closed Fist",
        "min_confidence": 0.7,
        "color": (231, 76, 60)  # Red
    },
    'Pointing_Up': {
        "display": "👆 Pointing Up",
        "min_confidence": 0.75,
        "color": (52, 152, 219)  # Blue
    },
    'Victory': {
        "display": "✌️ Victory",
        "min_confidence": 0.8,
        "color": (155, 89, 182)  # Purple
    },
    'Thumb_Up': {
        "display": "👍 Thumbs Up",
        "min_confidence": 0.8,
        "color": (241, 196, 15)  # Yellow
    },
    'ILoveYou': {
        "display": "🤟 I Love You",
        "min_confidence": 0.85,
        "color": (230, 126, 34)  # Orange
    }
}

def convert_to_landmark_list(hand_landmarks):
    """Convert MediaPipe Tasks format to standard MediaPipe format"""
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    for landmark in hand_landmarks:
        new_landmark = landmark_list.landmark.add()
        new_landmark.x = landmark.x
        new_landmark.y = landmark.y
        new_landmark.z = landmark.z
    return landmark_list

def process_frame(frame, recognizer, sensitivity, selected_gestures):
    """Process each frame for gesture detection"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    # Process with MediaPipe
    result = recognizer.recognize(mp_image)
    
    annotated_frame = frame.copy()
    current_gestures = []
    
    if result.gestures:
        for idx, gesture_list in enumerate(result.gestures):
            gesture = gesture_list[0]
            gesture_name = gesture.category_name
            
            # Only process configured gestures
            if gesture_name not in GESTURE_CONFIG:
                continue
                
            config = GESTURE_CONFIG[gesture_name]
            min_confidence = config["min_confidence"] * sensitivity
            
            if gesture.score >= min_confidence:
                display_name = config["display"]
                current_gestures.append(display_name)
                
                # Draw landmarks if hand detected
                if idx < len(result.hand_landmarks):
                    landmarks = result.hand_landmarks[idx]
                    landmark_list = convert_to_landmark_list(landmarks)
                    
                    # Draw with gesture-specific color
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        landmark_list,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=config["color"], thickness=4, circle_radius=6),
                        mp_drawing.DrawingSpec(color=config["color"], thickness=3, circle_radius=3)
                    )
                    
                    # Draw bounding box
                    x_coords = [l.x for l in landmarks]
                    y_coords = [l.y for l in landmarks]
                    h, w = annotated_frame.shape[:2]
                    x1, y1 = int(min(x_coords) * w), int(min(y_coords) * h)
                    x2, y2 = int(max(x_coords) * w), int(max(y_coords) * h)
                    
                    # Highlight selected gestures
                    border_color = config["color"]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), border_color, 3)
                    
                    # Draw gesture info
                    text = f"{display_name} ({gesture.score:.0%})"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(annotated_frame, 
                                 (x1-5, y1-text_height-15), 
                                 (x1+text_width+5, y1-5), 
                                 config["color"], -1)
                    cv2.putText(annotated_frame, text,
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                               (255, 255, 255), 2, cv2.LINE_AA)
    
    return annotated_frame, current_gestures

def main():
    st.title("✨ Professional Hand Gesture Recognition")
    st.markdown('<div class="header">Advanced Computer Vision Demonstration</div>', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        sensitivity = st.slider("Detection Sensitivity", 0.1, 1.0, 0.8, 0.05,
                               help="Adjust to fine-tune gesture recognition accuracy")
        
        selected_gestures = st.multiselect(
            "Focus Gestures",
            [config["display"] for config in GESTURE_CONFIG.values()],
            default=["🖐️ Open Palm", "✊ Closed Fist", "👆 Pointing Up"],
            help="Select which gestures to highlight with colors"
        )
        
        st.markdown("---")
        st.markdown("**Gesture Configuration**")
        for gesture, config in GESTURE_CONFIG.items():
            st.caption(f"{config['display']}: Min confidence {config['min_confidence']*100:.0f}%")
    
    # Initialize MediaPipe
    base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)
    
    # WebRTC streamer for better webcam handling
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img, current_gestures = process_frame(img, recognizer, sensitivity, selected_gestures)
        
        # Update session state
        if current_gestures:
            current_time = time.strftime("%H:%M:%S")
            if 'gesture_log' not in st.session_state:
                st.session_state.gesture_log = []
            
            # Only log if different from last logged gesture
            if not st.session_state.gesture_log or st.session_state.gesture_log[-1][0] != current_gestures[0]:
                st.session_state.gesture_log.append((current_gestures[0], current_time))
                st.session_state.gesture_log = st.session_state.gesture_log[-8:]  # Keep last 8
        
        return av.VideoFrame.from_ndarray(processed_img, format="rgb24")

    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Live Detection")
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        ctx = webrtc_streamer(
            key="gesture-recognition",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if not ctx.state.playing:
            st.info("Waiting for camera...")
            st.image("https://via.placeholder.com/640x360.png?text=Waiting+for+camera", 
                    use_container_width=True)
    
    with col2:
        st.markdown("### Detection Metrics")
        
        if 'gesture_log' in st.session_state and st.session_state.gesture_log:
            current_gesture = st.session_state.gesture_log[-1][0]
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Current Gesture</h3>
                    <h2>{current_gesture}</h2>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="metric-card">
                    <h3>Current Gesture</h3>
                    <h2>None detected</h2>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="metric-card">
                <h3>System Status</h3>
                <p>Camera: Active</p>
                <p>Detection: Running</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Recent Gestures")
        if 'gesture_log' in st.session_state and st.session_state.gesture_log:
            for gesture, timestamp in reversed(st.session_state.gesture_log):
                highlight_class = "highlight-badge" if gesture in selected_gestures else ""
                st.markdown(f'''
                    <div class="gesture-badge {highlight_class}">
                        {gesture} <small>({timestamp})</small>
                    </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("No gestures detected yet")

if __name__ == "__main__":
    main()
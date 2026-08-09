import streamlit as st
import numpy as np
import cv2
import av
from PIL import Image
import logging
import time
import collections
import threading
from typing import Optional, Any, Tuple

import config
import ui
from mediapipe_detector import PoseDetector

logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Fitness Assistant", layout="wide") 

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

@st.cache_resource
def get_ai_detector() -> Optional[PoseDetector]:
    """Safely initializes the pipeline outside the WebRTC worker context."""
    try:
        return PoseDetector()
    except Exception as e:
        logger.critical(f"App Initialization Failed: {e}", exc_info=True)
        return None

def prepare_frame(uploaded_file: Any) -> Optional[np.ndarray]:
    try:
        image_pil = Image.open(uploaded_file).convert("RGB")
        return np.array(image_pil)
    except Exception as e:
        logger.error(f"Image conversion failed: {e}")
        return None

def handle_input_stream(detector: Optional[PoseDetector]) -> Tuple[Optional[Any], str]:
    input_source = ui.render_sidebar(detector)
    
    if input_source == config.INPUT_SOURCE_UPLOAD:
        return ui.render_image_uploader(), input_source
    elif input_source == config.INPUT_SOURCE_CAMERA:
        return ui.render_camera_input(), input_source
    
    return None, input_source

def process_and_display(frame: np.ndarray, detector: PoseDetector) -> None:
    """Handles static image inputs cleanly with the unified V7 Dashboard."""
    with st.spinner("Analyzing pose..."):
        pose_landmarks = detector.extract_landmarks(frame, is_live_stream=False)
        prediction_result, confidence = detector.predict_frame(pose_landmarks)
        processed_frame = detector.draw_landmarks(frame.copy(), pose_landmarks)
        
    col_feed, col_dash = st.columns([7, 3])
    
    with col_feed:
        ui.render_images(frame, processed_frame)
        if not prediction_result:
            ui.render_warning("No human pose detected. Please ensure your full body is visible.")
            
    with col_dash:
        is_active = bool(prediction_result)
        ui.render_right_dashboard(
            prediction=prediction_result, 
            confidence=confidence, 
            is_active=is_active,
            fps=None,
            latency=None
        )

# ==========================================================
# LIVE STATE ARCHITECTURE (VERSION 7)
# ==========================================================

class LiveStateProcessor:
    """
    Thread-safe processor handling high-frequency WebRTC frames.
    Holds atomic snapshot data strictly for the Dashboard to consume at low-frequency.
    """
    def __init__(self, detector: PoseDetector):
        self.detector = detector
        self.lock = threading.Lock()
        
        # UI State Variables
        self.prediction: Optional[str] = None
        self.confidence: float = 0.0
        self.is_active: bool = False
        self.fps: float = 0.0
        self.latency: float = 0.0
        
        # Internal Pipeline Variables
        self._pred_history = collections.deque(maxlen=config.PREDICTION_HISTORY_SIZE)
        self._last_frame_time = time.perf_counter()

    def get_snapshot(self) -> dict:
        """Atomic read by the Main Thread to render the Dashboard safely."""
        with self.lock:
            return {
                "prediction": self.prediction,
                "confidence": self.confidence,
                "is_active": self.is_active,
                "fps": self.fps,
                "latency": self.latency
            }

    def video_frame_callback(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Isolated background worker loop (~30 FPS)."""
        start_process_time = time.perf_counter()
        
        try:
            image_bgr = frame.to_ndarray(format="bgr24")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Step 1: AI Inference (Unblocked)
            pose_landmarks = self.detector.extract_landmarks(image_rgb, is_live_stream=True)
            raw_pred, raw_conf = self.detector.predict_frame(pose_landmarks)
            
            # Step 2: Prediction Stabilization (Majority Voting)
            if raw_pred and raw_conf >= config.PREDICTION_CONFIDENCE_THRESHOLD:
                self._pred_history.append((raw_pred, raw_conf))
            else:
                self._pred_history.append((None, 0.0))
                
            valid_preds = [p[0] for p in self._pred_history if p[0] is not None]
            
            if valid_preds:
                most_common_pred = collections.Counter(valid_preds).most_common(1)[0][0]
                confs = [p[1] for p in self._pred_history if p[0] == most_common_pred]
                
                final_pred = most_common_pred
                final_conf = sum(confs) / len(confs)
                final_active = True
            else:
                final_pred = None
                final_conf = 0.0
                final_active = False
                
            # Step 3: Draw visual skeleton only
            processed_rgb = self.detector.draw_landmarks(image_rgb, pose_landmarks)
            processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
            
            # Step 4: Metric Calcs
            end_process_time = time.perf_counter()
            latency = (end_process_time - start_process_time) * 1000
            current_fps = 1.0 / (end_process_time - self._last_frame_time + 1e-6)
            self._last_frame_time = end_process_time

            # Step 5: Thread-Safe State Update (Microsecond lock duration)
            with self.lock:
                self.prediction = final_pred
                self.confidence = final_conf
                self.is_active = final_active
                # Exponential Moving Average for smooth dashboard visual metrics
                self.latency = (self.latency * 0.9) + (latency * 0.1) if self.latency > 0 else latency
                self.fps = (self.fps * 0.9) + (current_fps * 0.1) if self.fps > 0 else current_fps
                
            return av.VideoFrame.from_ndarray(processed_bgr, format="bgr24")
            
        except Exception as e:
            # Prevents a single faulty frame from crashing the ongoing RTC connection.
            logger.error("Live processing error", exc_info=True)
            return frame

# ==========================================================
# MAIN CONTROLLER
# ==========================================================

def main() -> None:
    detector = get_ai_detector()
    if detector is None:
        ui.render_error("Failed to load AI Models. Check server logs.")
        return

    ui.render_header()
    
    input_image, input_source = handle_input_stream(detector)
    
    if input_source == config.INPUT_SOURCE_LIVE:
        
        # Persist the processor so its history buffer and metrics survive st.rerun()
        if "live_processor" not in st.session_state:
            st.session_state.live_processor = LiveStateProcessor(detector)
        processor = st.session_state.live_processor
        
        col_feed, col_dash = st.columns([7, 3])
        
        with col_feed:
            ctx = ui.render_webrtc_player(processor.video_frame_callback)
            
        with col_dash:
            # Controlled Dashboard Update Loop (approx 1 Hz)
            if ctx and ctx.state.playing:
                snapshot = processor.get_snapshot()
                
                ui.render_right_dashboard(
                    prediction=snapshot["prediction"],
                    confidence=snapshot["confidence"],
                    is_active=snapshot["is_active"],
                    fps=snapshot["fps"],
                    latency=snapshot["latency"]
                )
                
                # Decouples the slow UI render cycle from the fast AI WebRTC cycle.
                time.sleep(1.0)
                st.rerun()
            else:
                ui.render_right_dashboard(is_active=False)
        return
        
    if input_image is not None:
        frame = prepare_frame(input_image)
        if frame is not None:
            process_and_display(frame, detector)
        else:
            ui.render_error("Could not read the provided image. Please try a different input.")
    else:
        col_feed, col_dash = st.columns([7, 3])
        with col_feed:
            st.info("Please provide an image input to begin analysis.")
        with col_dash:
            ui.render_right_dashboard(is_active=False)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical("Unhandled critical exception in main loop", exc_info=True)
        st.error("An unexpected application error occurred. Please contact support.")
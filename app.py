"""
Application Orchestration Module for AI Pose Pro.

Responsibilities:
    - Streamlit page and Session State initialization.
    - Instantiation of the AI Inference Engine exactly once.
    - WebRTC setup and connection to the PoseDetector.
    - Non-blocking UI synchronization using Streamlit Fragments.
    - Application routing and global component rendering.

This module operates as the final integration layer and strictly adheres 
to the official Streamlit Fragment architecture and streamlit-webrtc thread model.
"""

import logging
import queue
import time

import av
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase
)

# Strictly importing from approved modules
from config import (
    UI, 
    PREDICTIONS, 
    MODELS, 
    STATUS, 
    RTC_CONFIGURATION, 
    configure_logging
)
from mediapipe_detector import PoseDetector, PredictionOutput
import ui


# ==============================================================================
# LOGGING & RESOURCE MANAGEMENT
# ==============================================================================

logger = logging.getLogger("ai_pose_pro.app")

# Note: Explicit atexit cleanup is intentionally removed.
# The PoseDetector handles its own lifecycle and memory deallocation 
# natively via its __del__ and release() implementation.


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

def initialize_session_state() -> None:
    """
    Initializes required application state exactly once.
    Ensures the PoseDetector and communication queues are not repeatedly recreated.
    """
    if "logging_configured" not in st.session_state:
        configure_logging()
        st.session_state.logging_configured = True

    if "app_initialized" not in st.session_state:
        logger.info("Initializing application session state...")
        
        # Create the detector exactly once
        st.session_state.detector = PoseDetector()
        
        # Create a thread-safe queue for WebRTC -> UI synchronization
        st.session_state.result_queue = queue.Queue(maxsize=PREDICTIONS.queue_size)
        
        # Initialize with a valid default PredictionOutput to maintain UI layout
        st.session_state.last_prediction = PredictionOutput(
            predicted_label=PREDICTIONS.default_prediction_label,
            confidence=PREDICTIONS.default_confidence,
            model_name=MODELS.framework.value,
            prediction_status=STATUS.det_idle,
            fps=0.0,
            is_person_detected=False,
            timestamp=time.time()
        )
        
        # Mark initialization as complete
        st.session_state.app_initialized = True
        logger.info("Session initialized successfully.")


# ==============================================================================
# WEBRTC VIDEO PROCESSOR
# ==============================================================================

class VideoProcessor(VideoProcessorBase):
    """
    Dedicated VideoProcessor class for streamlit-webrtc.
    Executes in a background worker thread.
    
    Responsibilities:
        - Receives live frames from the WebRTC stream.
        - Passes frames to the AI inference engine.
        - Pushes the latest PredictionOutput into the queue.
        - Returns the annotated frame back to the browser.
        - Contains ZERO Streamlit API calls.
    """
    def __init__(self, detector: PoseDetector, result_queue: queue.Queue) -> None:
        self.detector: PoseDetector = detector
        self.queue: queue.Queue = result_queue

    def _push_latest_prediction(self, prediction: PredictionOutput) -> None:
        """
        Safely enforces maxsize queue logic for real-time synchronization.
        """
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
                
        try:
            self.queue.put_nowait(prediction)
        except queue.Full:
            pass

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Processes an incoming video frame and synchronizes the prediction."""
        # Convert incoming frame strictly to RGB24 per contract
        img = frame.to_ndarray(format="rgb24")
        
        # AI Inference
        prediction: PredictionOutput = self.detector.process_frame(img)
        
        # Synchronize prediction with the main Streamlit thread
        self._push_latest_prediction(prediction)
            
        # Return annotated RGB24 frame
        return av.VideoFrame.from_ndarray(img, format="rgb24")


# ==============================================================================
# UI DASHBOARD FRAGMENT
# ==============================================================================

@st.fragment(run_every="0.1s")
def render_dashboard_fragment() -> None:
    """
    Isolated execution context for the right-side dashboard.
    Polls the thread-safe queue and pushes updates without causing full script reruns.
    """
    # Guard against missing session_state keys during early initialization
    if "result_queue" not in st.session_state or "last_prediction" not in st.session_state:
        return

    try:
        prediction = st.session_state.result_queue.get_nowait()
        st.session_state.last_prediction = prediction
    except queue.Empty:
        pass
        
    ui.render_dashboard(st.session_state.last_prediction)


# ==============================================================================
# MAIN APPLICATION ORCHESTRATION
# ==============================================================================

def main() -> None:
    """
    Primary entry point for the Streamlit application.
    Orchestrates the entire layout using strict UI boundaries and fragment updates.
    """
    # 1. Page Configuration (Must be the first Streamlit command executed)
    st.set_page_config(
        page_title=UI.page_title,
        page_icon=UI.page_icon,
        layout=UI.layout,
        initial_sidebar_state=UI.initial_sidebar_state
    )

    # 2. State Initialization
    initialize_session_state()
    
    # 3. Global CSS Injection
    ui.inject_global_css()
    
    # 4. Global Header Render
    ui.render_header()
    
    # 5. Sidebar Navigation Render
    selected_page = ui.render_sidebar()
    
    # 6. Routing & Content Generation
    if selected_page == UI.about_title:
        ui.render_about_page()
        
    elif selected_page == UI.dashboard_title:
        left_col, right_col = ui.create_main_layout()
        
        # --- LEFT COLUMN: LIVE WEBCAM ---
        with left_col:
            st.markdown(f"<div class='ui-header'>{UI.camera_title}</div>", unsafe_allow_html=True)
            
            try:
                # Capture session state variables for thread-safe dependency injection
                active_detector = st.session_state.detector
                active_queue = st.session_state.result_queue
                
                # Factory function passes shared state cleanly to the worker thread
                def video_processor_factory() -> VideoProcessor:
                    return VideoProcessor(detector=active_detector, result_queue=active_queue)

                webrtc_streamer(
                    key=UI.webrtc_key,
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=video_processor_factory,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
            except Exception:
                logger.exception("WebRTC initialization failure.")
                st.error("Camera stream is currently unavailable. Please check your device permissions or network connection.")
        
        # --- RIGHT COLUMN: DASHBOARD FRAGMENT ---
        with right_col:
            render_dashboard_fragment()

    # 7. Global Footer Render
    ui.render_footer()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
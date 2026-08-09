import numpy as np
import mediapipe as mp
import tensorflow as tf
import cv2
import joblib
import logging
import os
import warnings
from typing import Tuple, Optional, Any

import config

logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)
logger = logging.getLogger(__name__)

class PoseDetector:
    """
    Core AI Pipeline Controller.
    Responsibility: Provides modular, single-responsibility functions for 
    landmark extraction, feature scaling, prediction, and visual overlays.
    """
    
    def __init__(self) -> None:
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self._pose_static = None
        self._pose_stream = None
        
        self._load_models()

    def _get_mediapipe_instance(self, is_live_stream: bool) -> Any:
        try:
            if is_live_stream:
                if self._pose_stream is None:
                    self._pose_stream = self.mp_pose.Pose(
                        static_image_mode=False,
                        model_complexity=config.MP_MODEL_COMPLEXITY,
                        min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE
                    )
                    logger.info("MediaPipe initialized in CONTINUOUS STREAM mode.")
                return self._pose_stream
            else:
                if self._pose_static is None:
                    self._pose_static = self.mp_pose.Pose(
                        static_image_mode=True,
                        model_complexity=config.MP_MODEL_COMPLEXITY,
                        min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE
                    )
                    logger.info("MediaPipe initialized in STATIC IMAGE mode.")
                return self._pose_static
        except Exception as e:
            logger.critical(f"{config.ERR_MEDIAPIPE_FAILED} Details: {e}")
            raise RuntimeError("MediaPipe initialization failed.")

    def _load_models(self) -> None:
        if not os.path.exists(config.MODEL_PATH):
            logger.critical(config.ERR_MODEL_NOT_FOUND)
            raise FileNotFoundError(config.ERR_MODEL_NOT_FOUND)

        try:
            self.model = tf.keras.models.load_model(config.MODEL_PATH)
            self.scaler = joblib.load(config.SCALER_PATH)
            self.label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
            logger.info("AI Model artifacts loaded successfully.")
        except Exception as e:
            logger.error(f"Unexpected error loading models: {e}")
            raise

    def extract_landmarks(self, frame: np.ndarray, is_live_stream: bool = False) -> Any:
        active_pose_processor = self._get_mediapipe_instance(is_live_stream)
        results = active_pose_processor.process(frame)
        return results.pose_landmarks

    def predict_frame(self, pose_landmarks: Any) -> Tuple[Optional[str], Optional[float]]:
        if not pose_landmarks:
            return None, None
            
        features = []
        for landmark in pose_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        raw_features = np.array(features).reshape(1, -1)
        
        # Safely suppress sklearn missing-feature-names warning without modifying metadata
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            scaled_features = self.scaler.transform(raw_features)
            
        predictions = self.model.predict(scaled_features, verbose=0)
        
        predicted_class_index = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))
        prediction_label = self.label_encoder.inverse_transform([predicted_class_index])[0]
        
        return prediction_label, confidence

    def draw_landmarks(self, frame: np.ndarray, pose_landmarks: Any) -> np.ndarray:
        if pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame

    def render_prediction_overlay(self, frame: np.ndarray, prediction: Optional[str], confidence: Optional[float]) -> np.ndarray:
        """
        No-op in Version 7. Camera feed must strictly display only the video and skeleton.
        """
        return frame
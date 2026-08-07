"""
AI Inference Engine Module for AI Pose Pro.

Responsibilities:
    - Handles the complete computer vision and AI prediction pipeline.
    - Manages MediaPipe Pose initialization, human pose detection, and landmark validation.
    - Responsible for feature extraction, scaling, and Artificial Neural Network inference.

Thread ownership:
    - This module is designed to run in a background worker thread (e.g., via WebRTC).
    - It contains absolutely no UI, Streamlit, session state, or application routing logic.

Frame format contract:
    - ZERO AMBIGUITY: This engine STRICTLY expects RGB video frames.
    - The caller (Application Layer) MUST convert BGR or other formats to RGB 
      before passing the frame to `process_frame()`. 
    - The inference engine does NOT perform color space conversions internally.

PredictionOutput contract:
    - Exclusively returns a single, immutable `PredictionOutput` dataclass per frame.
    - Guarantees a valid, predictable return object even during critical failures.

Lifecycle:
    - Requires instantiation once. Models and ML artifacts are loaded exactly once.
    - The `release()` method safely deallocates MediaPipe and ML resources.
    - Automatic cleanup is supported via `__del__()`.

This is the designated FINAL FREEZE VERSION of the inference engine.
"""

import time
import logging
import pickle
import joblib
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Any, Optional

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from config import (
    PATHS,
    MODELS,
    PREDICTIONS,
    MEDIAPIPE,
    STATUS
)


# ==============================================================================
# DATA CONTRACT
# ==============================================================================

@dataclass(frozen=True)
class PredictionOutput:
    """
    Immutable data contract representing a single frame's inference result.
    This is the ONLY object returned by the inference engine to the application layer.
    """
    predicted_label: str
    confidence: float
    model_name: str
    prediction_status: str
    fps: float
    is_person_detected: bool
    timestamp: float


# ==============================================================================
# INFERENCE ENGINE
# ==============================================================================

class PoseDetector:
    """
    Production-grade AI Inference Engine for real-time exercise recognition.
    
    Handles the entire lifecycle of computer vision and machine learning inference.
    STRICT CONTRACT: All incoming frames must be in RGB format.
    """

    def __init__(self) -> None:
        """Initializes the inference engine, ML models, and state tracking."""
        self._logger = logging.getLogger("ai_pose_pro.detector")
        self._logger.info("Initializing PoseDetector Inference Engine...")
        
        # System State
        self._models_loaded: bool = False
        self._model: Optional[tf.keras.Model] = None
        self._scaler: Any = None
        self._label_encoder: Any = None
        self._last_error: Optional[str] = None
        
        # MediaPipe Components
        self._mp_pose: Any = None
        self._mp_drawing: Any = None
        self._pose: Any = None
        
        # Tracking & Smoothing State
        self._prev_time: float = time.time()
        self._ema_fps: float = 0.0
        self._fps_alpha: float = 0.1  # Smoothing factor for EMA
        self._prediction_queue: deque = deque(maxlen=PREDICTIONS.smoothing_window_size)
        
        # Boot Sequence
        self._initialize_pose()
        self._load_models()

    def __del__(self) -> None:
        """Safely cleans up resources upon object destruction."""
        try:
            self.release()
        except Exception:
            pass

    def _initialize_pose(self) -> None:
        """Privately initializes the MediaPipe Pose tracking solution."""
        try:
            self._mp_pose = mp.solutions.pose
            self._mp_drawing = mp.solutions.drawing_utils
            
            self._pose = self._mp_pose.Pose(
                static_image_mode=MEDIAPIPE.static_image_mode,
                model_complexity=MEDIAPIPE.model_complexity,
                smooth_landmarks=MEDIAPIPE.smooth_landmarks,
                enable_segmentation=MEDIAPIPE.enable_segmentation,
                smooth_segmentation=MEDIAPIPE.smooth_segmentation,
                min_detection_confidence=MEDIAPIPE.min_detection_confidence,
                min_tracking_confidence=MEDIAPIPE.min_tracking_confidence
            )
            self._logger.info("MediaPipe Pose engine initialized and ready.")
        except Exception as e:
            self._last_error = f"MediaPipe initialization failed: {str(e)}"
            self._logger.exception("Critical failure during MediaPipe initialization.")
            raise

    def _load_models(self) -> None:
        """
        Privately loads and validates the pre-trained ANN model, StandardScaler, 
        and LabelEncoder exactly once into memory.
        """
        try:
            # 1. Load and Validate ANN
            self._logger.info(f"Loading ANN model from {PATHS.ann_model_path}")
            self._model = tf.keras.models.load_model(str(PATHS.ann_model_path))
            
            input_shape = self._model.input_shape[-1]
            output_shape = self._model.output_shape[-1]
            
            if input_shape != MODELS.input_features_count:
                raise ValueError(f"ANN input mismatch: expected {MODELS.input_features_count}, got {input_shape}")
            if output_shape != MODELS.output_class_count:
                raise ValueError(f"ANN output mismatch: expected {MODELS.output_class_count}, got {output_shape}")
            
            # 2. Load and Validate Scaler
            self._logger.info(f"Loading StandardScaler from {PATHS.scaler_path}")
            self._scaler = joblib.load(PATHS.scaler_path)
                
            if getattr(self._scaler, 'n_features_in_', None) != MODELS.input_features_count:
                raise ValueError("Scaler expected feature dimension mismatch.")
                
            # 3. Load and Validate LabelEncoder
            self._logger.info(f"Loading LabelEncoder from {PATHS.label_encoder_path}")
            self._label_encoder = joblib.load(PATHS.label_encoder_path)
                
            if len(self._label_encoder.classes_) != MODELS.output_class_count:
                raise ValueError("LabelEncoder class count mismatch with ANN output.")
                
            self._models_loaded = True
            self._logger.info("All machine learning artifacts loaded and validated successfully.")
            
        except Exception as e:
            self._last_error = f"Machine learning artifact validation/loading failed: {str(e)}"
            self._logger.exception("Failed to load or validate machine learning artifacts.")
            self._models_loaded = False

    def is_ready(self) -> bool:
        """
        Validates the overall readiness of the inference engine.
        
        Returns:
            bool: True if MediaPipe and ML models are fully initialized and validated.
        """
        return self._models_loaded and self._pose is not None

    def release(self) -> None:
        """Safely releases MediaPipe resources and clears ML artifacts from memory."""
        self._logger.info("Releasing PoseDetector resources...")
        if self._pose:
            self._pose.close()
            self._pose = None
        
        self._model = None
        self._scaler = None
        self._label_encoder = None
        self._models_loaded = False
        self._reset_state()
        self._logger.info("PoseDetector resources released successfully.")

    def _reset_state(self) -> None:
        """Resets tracking variables, FPS calculations, and prediction history."""
        self._prev_time = time.time()
        self._ema_fps = 0.0
        self._clear_prediction_history()

    def _clear_prediction_history(self) -> None:
        """Clears the temporal smoothing queue for predictions."""
        self._prediction_queue.clear()

    def _calculate_fps(self) -> float:
        """
        Calculates and smooths Frames Per Second (FPS) using an Exponential Moving Average.
        
        Returns:
            float: Current smoothed FPS, rounded to 2 decimal places.
        """
        current_time = time.time()
        dt = current_time - self._prev_time
        self._prev_time = current_time
        
        if dt > 0:
            current_fps = 1.0 / dt
            if self._ema_fps == 0.0:
                self._ema_fps = current_fps
            else:
                self._ema_fps = (self._fps_alpha * current_fps) + ((1.0 - self._fps_alpha) * self._ema_fps)
                
        return round(self._ema_fps, 2)

    def _validate_landmarks(self, pose_landmarks: Any) -> bool:
        """
        Validates if the detected pose has enough visible landmarks to qualify for inference.
        
        Args:
            pose_landmarks: MediaPipe normalized landmark list.
            
        Returns:
            bool: True if landmark visibility meets production thresholds, False otherwise.
        """
        visible_count = sum(
            1 for lm in pose_landmarks.landmark 
            if lm.visibility >= MEDIAPIPE.min_detection_confidence
        )
        return visible_count >= PREDICTIONS.min_visible_landmarks

    def _extract_features(self, pose_landmarks: Any) -> np.ndarray:
        """
        Extracts all 33 pose landmarks and generates exactly 132 numerical features.
        The layout for each landmark is [x, y, z, visibility].
        Pre-allocates the numpy array to optimize real-time streaming performance.
        
        Args:
            pose_landmarks: MediaPipe normalized landmark list.
            
        Returns:
            np.ndarray: A 1x132 dimensional array for ANN inference.
        """
        features = np.empty((1, MODELS.input_features_count), dtype=np.float32)
        idx = 0
        for lm in pose_landmarks.landmark:
            features[0, idx] = lm.x
            features[0, idx+1] = lm.y
            features[0, idx+2] = lm.z
            features[0, idx+3] = lm.visibility
            idx += 4
            
        return features

    def _smooth_prediction(self, label: str, confidence: float) -> Tuple[str, float]:
        """
        Applies temporal smoothing to predictions to prevent rapid flickering.
        
        Args:
            label: The current frame's predicted string label.
            confidence: The current frame's raw prediction probability.
            
        Returns:
            Tuple[str, float]: The smoothed label and normalized/rounded confidence score.
        """
        self._prediction_queue.append((label, confidence))
        
        # Calculate mode for labels
        labels = [p[0] for p in self._prediction_queue]
        most_common_label = max(set(labels), key=labels.count)
        
        # Average the confidence associated with the most common label
        confidences = [p[1] for p in self._prediction_queue if p[0] == most_common_label]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return most_common_label, round(avg_confidence, 4)

    def _predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Runs the full inference pipeline (Scaling -> ANN Forward Pass -> Decoding -> Smoothing).
        Includes strict probabilistic validation logic to prevent silent calculation failures.
        
        Args:
            features: 1x132 numpy array representing the pose.
            
        Returns:
            Tuple[str, float]: The final predicted class label and its normalized confidence.
        """
        try:
            # Preprocessing
            scaled_features = self._scaler.transform(features)
            
            # Inference pass
            probabilities = self._model.predict(scaled_features, verbose=0)[0]
            
            # Prediction Integrity Validation
            if probabilities.size == 0 or np.isnan(probabilities).any() or np.isinf(probabilities).any():
                self._last_error = "ANN returned NaN/Inf or empty probability vector."
                self._logger.warning(self._last_error)
                return STATUS.pred_unknown, 0.0
                
            max_index = int(np.argmax(probabilities))
            raw_confidence = float(probabilities[max_index])
            
            # Postprocessing & Thresholding
            if raw_confidence >= PREDICTIONS.confidence_threshold:
                raw_label = str(self._label_encoder.inverse_transform([max_index])[0])
            else:
                raw_label = STATUS.pred_unknown
                
            return self._smooth_prediction(raw_label, raw_confidence)
            
        except Exception as e:
            self._last_error = f"Prediction inference failed unexpectedly: {str(e)}"
            self._logger.exception(self._last_error)
            return STATUS.sys_error, 0.0

    def _draw_landmarks(self, frame: np.ndarray, results: Any) -> None:
        """
        Mutates the incoming frame array to draw tracking visualizations.
        Expects an RGB frame per the module contract.
        
        Args:
            frame: Numpy array representing the current video frame (RGB).
            results: MediaPipe processing results containing landmarks.
        """
        if not MEDIAPIPE.draw_landmarks or not results.pose_landmarks:
            return
            
        try:
            self._mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.pose_landmarks,
                connections=self._mp_pose.POSE_CONNECTIONS if MEDIAPIPE.draw_connections else None,
                landmark_drawing_spec=self._mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=MEDIAPIPE.landmark_thickness,
                    circle_radius=MEDIAPIPE.circle_radius
                ),
                connection_drawing_spec=self._mp_drawing.DrawingSpec(
                    color=(255, 255, 255),
                    thickness=MEDIAPIPE.connection_thickness
                )
            )
        except Exception as e:
            self._last_error = f"Failed to draw pose annotations: {str(e)}"
            self._logger.exception("Failed to draw pose annotations.")

    def process_frame(self, frame: np.ndarray) -> PredictionOutput:
        """
        The primary public interface for the inference engine.
        Accepts a video frame, mutates it in place with annotations if enabled, 
        and extracts business intelligence as a PredictionOutput object.
        
        FRAME FORMAT CONTRACT:
        This method STRICTLY expects an RGB video frame. It will not perform BGR to RGB 
        conversions internally. 
        
        Args:
            frame: The current numpy video frame in RGB format.
            
        Returns:
            PredictionOutput: The immutable data contract representing the system state.
        """
        fps = self._calculate_fps()
        timestamp = time.time()
        default_conf = round(PREDICTIONS.default_confidence, 4)
        
        # 1. Integrity Check
        if frame is None or frame.size == 0:
            self._last_error = "Received empty frame payload."
            self._logger.warning(self._last_error)
            return PredictionOutput(
                predicted_label=PREDICTIONS.default_prediction_label,
                confidence=default_conf,
                model_name=MODELS.framework.value,
                prediction_status=STATUS.camera_error,
                fps=fps,
                is_person_detected=False,
                timestamp=timestamp
            )

        # 2. System State Check
        if not self.is_ready():
            return PredictionOutput(
                predicted_label=PREDICTIONS.default_prediction_label,
                confidence=default_conf,
                model_name=MODELS.framework.value,
                prediction_status=STATUS.model_loading,
                fps=fps,
                is_person_detected=False,
                timestamp=timestamp
            )

        # 3. MediaPipe Inference (Memory Optimized, Assumes RGB input)
        frame.flags.writeable = False
        try:
            results = self._pose.process(frame)
            print(results.pose_landmarks)
        except Exception as e:
            self._last_error = f"MediaPipe processing engine fault: {str(e)}"
            self._logger.exception("MediaPipe processing engine fault.")
            frame.flags.writeable = True
            return PredictionOutput(
                predicted_label=PREDICTIONS.default_prediction_label,
                confidence=default_conf,
                model_name=MODELS.framework.value,
                prediction_status=STATUS.sys_error,
                fps=fps,
                is_person_detected=False,
                timestamp=timestamp
            )
            
        frame.flags.writeable = True

        # 4. Annotation Rendering
        if results.pose_landmarks:
            self._draw_landmarks(frame, results)

        # 5. Pipeline Validation
        if not results.pose_landmarks:
            self._clear_prediction_history()
            return PredictionOutput(
                predicted_label=PREDICTIONS.default_prediction_label,
                confidence=default_conf,
                model_name=MODELS.framework.value,
                prediction_status=STATUS.det_no_person,
                fps=fps,
                is_person_detected=False,
                timestamp=timestamp
            )
            
        if not self._validate_landmarks(results.pose_landmarks):
            self._clear_prediction_history()
            return PredictionOutput(
                predicted_label=PREDICTIONS.default_prediction_label,
                confidence=default_conf,
                model_name=MODELS.framework.value,
                prediction_status=STATUS.det_lost,
                fps=fps,
                is_person_detected=True,
                timestamp=timestamp
            )
            
        # 6. Feature Extraction & Validation
        features = self._extract_features(results.pose_landmarks)
        print("Features:", features.shape)
        
        if features.shape[1] != MODELS.input_features_count:
            self._last_error = f"Feature dimensionality mismatch. Expected {MODELS.input_features_count}, got {features.shape[1]}"
            self._logger.warning(self._last_error)
            return PredictionOutput(
                predicted_label=PREDICTIONS.default_prediction_label,
                confidence=default_conf,
                model_name=MODELS.framework.value,
                prediction_status=STATUS.sys_warning,
                fps=fps,
                is_person_detected=True,
                timestamp=timestamp
            )
            
        # 7. AI Inference
        label, confidence = self._predict(features)
        print(label)
        print(confidence)
        
        # Determine Prediction Stage Status
        if confidence >= PREDICTIONS.confidence_threshold:
            current_status = STATUS.prediction_running
        else:
            current_status = STATUS.pred_calculating
        
        # 8. Return Immutable Contract
        return PredictionOutput(
            predicted_label=label,
            confidence=confidence,
            model_name=MODELS.framework.value,
            prediction_status=current_status,
            fps=fps,
            is_person_detected=True,
            timestamp=timestamp
        )

print("Methods in PoseDetector:")
print([m for m in dir(PoseDetector) if m.startswith("_")])
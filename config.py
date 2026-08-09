import os
import logging

# ==========================================================
# PATH CONFIGURATIONS
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

MODEL_PATH = os.path.join(MODELS_DIR, 'exercise_ann.keras')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler_exercise.pkl')
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder_exercise.pkl')

# ==========================================================
# MEDIAPIPE INITIALIZATION PARAMETERS
# ==========================================================
MP_MODEL_COMPLEXITY = 2
MP_MIN_DETECTION_CONFIDENCE = 0.5

# ==========================================================
# LIVE PIPELINE CONFIGURATION (VERSION 7)
# ==========================================================
PREDICTION_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to accept a pose
PREDICTION_HISTORY_SIZE = 5            # Number of frames for majority voting

# ==========================================================
# UI & INPUT CONSTANTS
# ==========================================================
SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png']

INPUT_SOURCE_UPLOAD = "Upload Image"
INPUT_SOURCE_CAMERA = "Capture From Camera"
INPUT_SOURCE_LIVE = "Live Camera"
INPUT_SOURCES = [INPUT_SOURCE_UPLOAD, INPUT_SOURCE_CAMERA, INPUT_SOURCE_LIVE]

# ==========================================================
# ERROR MESSAGES
# ==========================================================
ERR_MODEL_NOT_FOUND = "Critical: AI model artifacts not found. Please check the models directory."
ERR_MEDIAPIPE_FAILED = "Error during MediaPipe landmark extraction."
ERR_PREDICTION_FAILED = "Error during ANN prediction pipeline."
ERR_IMAGE_LOAD = "Failed to load the input image. It might be corrupted or unsupported."
ERR_WEBRTC_FAILED = "Error processing live video stream."

# ==========================================================
# LOGGING CONFIGURATION
# ==========================================================
LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGING_LEVEL = logging.INFO
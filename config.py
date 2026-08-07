"""
Configuration module for AI Pose Pro.

This module provides the centralized, production-grade configuration layer 
for the entire application. It defines application metadata, file paths, 
machine learning settings, camera properties, UI theme details, and logging.

It contains no executable business logic, AI inference code, or UI rendering.
All constants and settings should be accessed globally through the exported
dataclass instances.

This is the designated FINAL FREEZE VERSION of the configuration layer.
"""

import logging
import sys
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple
from streamlit_webrtc import RTCConfiguration




RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)

# ==============================================================================
# BASE CONSTANTS
# ==============================================================================

_BASE_DIR = Path(__file__).parent.resolve()


# ==============================================================================
# ENUMS
# ==============================================================================

class SystemStatus(Enum):
    """Enumeration of overall system health states."""
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    WARNING = "WARNING"
    ERROR = "ERROR"


class DetectionState(Enum):
    """Enumeration of active detection states."""
    IDLE = "IDLE"
    DETECTING = "DETECTING"
    TRACKING_LOST = "TRACKING_LOST"
    NO_PERSON = "NO_PERSON"


class LogLevel(Enum):
    """Enumeration of supported logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelFramework(Enum):
    """Enumeration of supported AI inference frameworks."""
    TENSORFLOW = "TensorFlow"
    ONNX = "ONNX"
    TENSORRT = "TensorRT"


class ModelType(Enum):
    """Enumeration of model architectures."""
    ARTIFICIAL_NEURAL_NETWORK = "Artificial Neural Network"
    RANDOM_FOREST = "Random Forest"
    SVM = "Support Vector Machine"


class FrameFormat(Enum):
    """Enumeration of supported video frame formats."""
    RGB24 = "rgb24"
    BGR24 = "bgr24"


# ==============================================================================
# DATACLASSES
# ==============================================================================

@dataclass(frozen=True)
class AppConfig:
    """Application metadata and general settings."""
    name: str = "AI Pose Pro"
    description: str = "Real-Time Exercise & Yoga Pose Recognition System"
    author: str = "Production Development Team"
    
    # Semantic Versioning
    version_major: int = 1
    version_minor: int = 0
    version_patch: int = 0
    
    # Metadata
    github_url: str = "https://github.com/placeholder/ai-pose-pro"
    license: str = "MIT License"
    supported_python: str = ">=3.11"
    supported_streamlit: str = ">=1.20.0"
    
    technology_stack: Tuple[str, ...] = (
        "Python 3.11+", "Streamlit", "streamlit-webrtc", "MediaPipe Pose", 
        "TensorFlow 2.16.1", "OpenCV", "Scikit-Learn"
    )

    @property
    def full_version(self) -> str:
        """Returns the full semantic version string."""
        return f"{self.version_major}.{self.version_minor}.{self.version_patch}"


@dataclass(frozen=True)
class PathConfig:
    """Centralized directory and file paths."""
    base_dir: Path = field(default_factory=lambda: _BASE_DIR)
    
    # Core Directories
    models_dir: Path = field(default_factory=lambda: _BASE_DIR / "models")
    assets_dir: Path = field(default_factory=lambda: _BASE_DIR / "assets")
    logs_dir: Path = field(default_factory=lambda: _BASE_DIR / "logs")
    docs_dir: Path = field(default_factory=lambda: _BASE_DIR / "docs")
    temp_dir: Path = field(default_factory=lambda: _BASE_DIR / "temp")
    
    # Specific Files
    ann_model_path: Path = field(default_factory=lambda: _BASE_DIR / "models" /"exercise_ann.keras")
    label_encoder_path: Path = field(default_factory=lambda: _BASE_DIR / "models" / "label_encoder_exercise.pkl")
    scaler_path: Path = field(default_factory=lambda: _BASE_DIR / "models" / "scaler_exercise.pkl")
    logo_path: Path = field(default_factory=lambda: _BASE_DIR / "assets" / "logo.png")


@dataclass(frozen=True)
class ModelConfig:
    """Artificial Neural Network model metadata and structure."""
    framework: ModelFramework = ModelFramework.TENSORFLOW
    model_type: ModelType = ModelType.ARTIFICIAL_NEURAL_NETWORK
    version: str = "1.0"
    
    # Neural Network Shape
    input_features_count: int = 132
    output_class_count: int = 9
    
    # Version 1.0 Supported Exercise Classes
    supported_classes: Tuple[str, ...] = (
        "Battle Rope",
        "Bench Pressing",
        "Front Raise",
        "Jump Jack",
        "Pommel Horse",
        "Pull Up",
        "Push Up",
        "Sit Up",
        "Squat"
    )


@dataclass(frozen=True)
class PredictionConfig:
    """Inference boundaries, thresholds, and smoothing logic."""
    confidence_threshold: float = 0.70
    smoothing_window_size: int = 5          # Number of frames to average
    min_visible_landmarks: int = 15         # Minimum landmarks to accept a pose
    queue_size: int = 100                   # WebRTC async queue size
    prediction_timeout_ms: int = 5000       # Drop prediction if stale
    
    # Default UI Values
    default_prediction_label: str = "Awaiting input..."
    default_confidence: float = 0.0
    default_status: str = "INITIALIZING"


@dataclass(frozen=True)
class RuntimeConfig:
    """Application runtime flags and toggleable features."""
    debug_mode: bool = False
    production_mode: bool = True
    enable_logging: bool = True
    enable_fps: bool = True
    enable_landmarks: bool = True
    enable_confidence_display: bool = True


@dataclass(frozen=True)
class CameraConfig:
    """Webcam and video streaming configuration."""
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    mirror_feed: bool = True
    flip_feed: bool = False
    frame_format: FrameFormat = FrameFormat.RGB24
    webrtc_mode: str = "SENDRECV"


@dataclass(frozen=True)
class MediaPipeConfig:
    """MediaPipe Pose detection settings and rendering options."""
    # Engine Settings
    static_image_mode: bool = False
    model_complexity: int = 1
    smooth_landmarks: bool = True
    enable_segmentation: bool = False
    smooth_segmentation: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Rendering Settings
    draw_landmarks: bool = True
    draw_connections: bool = True
    landmark_thickness: int = 2
    connection_thickness: int = 2
    circle_radius: int = 2


@dataclass(frozen=True)
class ColorConfig:
    """Theme color palette based on UI Design System."""
    primary_bg: str = "#0B1120"      # Deep Navy Blue
    secondary_bg: str = "#1E293B"    # Dark Charcoal
    accent: str = "#3B82F6"          # Electric Blue
    success: str = "#10B981"         # Emerald Green
    warning: str = "#F59E0B"         # Amber
    error: str = "#EF4444"           # Crimson Red
    text_main: str = "#FFFFFF"       # White
    text_muted: str = "#9CA3AF"      # Light Gray


@dataclass(frozen=True)
class TypographyConfig:
    """Typography settings for the application."""
    font_family: str = "'Inter', 'Poppins', sans-serif"
    size_title: str = "2rem"
    size_header: str = "1.5rem"
    size_card_title: str = "1.25rem"
    size_metric: str = "2.5rem"
    size_label: str = "1rem"
    size_small: str = "0.875rem"


@dataclass(frozen=True)
class LayoutConfig:
    """Structural layout rules for UI responsiveness."""
    # Ratios (out of total width/columns)
    sidebar_ratio: int = 1
    camera_ratio: int = 6
    dashboard_ratio: int = 3
    
    # Spacing
    spacing_small: str = "0.5rem"
    spacing_medium: str = "1rem"
    spacing_large: str = "2rem"
    
    # Padding
    padding_small: str = "8px"
    padding_medium: str = "16px"
    padding_large: str = "24px"
    
    # Border
    border_radius_small: str = "8px"
    border_radius_medium: str = "12px"
    border_radius_large: str = "16px"


@dataclass(frozen=True)
class AnimationConfig:
    """CSS animation parameters and Glassmorphism settings."""
    transition_duration_ms: int = 300
    hover_duration_ms: int = 200
    
    # Glassmorphism
    glass_blur: str = "10px"
    glass_opacity: str = "0.7"
    glass_bg_rgba: str = "rgba(30, 41, 59, 0.7)"
    glass_border: str = "1px solid rgba(255, 255, 255, 0.1)"
    
    # Shadows
    shadow_light: str = "0 2px 4px -1px rgba(0, 0, 0, 0.06)"
    shadow_medium: str = "0 4px 6px -1px rgba(0, 0, 0, 0.1)"
    shadow_large: str = "0 10px 15px -3px rgba(0, 0, 0, 0.1)"


@dataclass(frozen=True)
class ThemeConfig:
    """Master theme configuration encapsulating visual system specifications."""
    css_transition: str = "all 0.3s ease-in-out"
    colors: ColorConfig = field(default_factory=ColorConfig)
    typography: TypographyConfig = field(default_factory=TypographyConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    animation: AnimationConfig = field(default_factory=AnimationConfig)


@dataclass(frozen=True)
class StatusConfig:
    """Centralized strings for internal status states to avoid magic strings."""
    # System Status Labels
    sys_initializing: str = "System Initializing..."
    sys_ready: str = "System Ready"
    sys_warning: str = "System Warning"
    sys_error: str = "Critical Error Encountered"
    
    # Subsystem States
    camera_ready: str = "Camera Ready"
    camera_error: str = "Camera Error"
    model_loading: str = "Loading Models..."
    model_loaded: str = "Models Loaded"
    prediction_running: str = "Prediction Running"
    prediction_idle: str = "Prediction Idle"
    
    # Detection State Labels
    det_idle: str = "Awaiting Camera Feed..."
    det_detecting: str = "Person Detected"
    det_lost: str = "Tracking Lost - Please return to frame"
    det_no_person: str = "No Person Visible"
    
    # Inference State Labels
    pred_calculating: str = "Calculating Confidence..."
    pred_unknown: str = "Unknown Pose"


@dataclass(frozen=True)
class LoggingConfig:
    """Structured logging configuration settings."""
    log_level: str = LogLevel.INFO.value
    log_format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class UIConfig:
    """User Interface specific string constants and setup configurations."""

    # Setup
    page_title: str = "AI Pose Pro"
    page_icon: str = "🤖"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"

    # Strings
    app_title: str = "AI Pose Pro Dashboard"
    sidebar_title: str = "Navigation"
    dashboard_title: str = "Live Dashboard"
    about_title: str = "About AI Pose Pro"

    # Camera
    camera_title: str = "Live Camera Feed"

    # WebRTC
    webrtc_key: str = "pose-recognition"

    metrics_title: str = "Live Performance Metrics"
    health_title: str = "System Health"

# ==============================================================================
# GLOBAL CONFIGURATION INSTANCES
# ==============================================================================

APP = AppConfig()
PATHS = PathConfig()
MODELS = ModelConfig()
PREDICTIONS = PredictionConfig()
RUNTIME = RuntimeConfig()
CAMERA = CameraConfig()
MEDIAPIPE = MediaPipeConfig()
THEME = ThemeConfig()
STATUS = StatusConfig()
LOGGING = LoggingConfig()
UI = UIConfig()

# ==============================================================================
# LOGGING SETUP UTILITY
# ==============================================================================

def configure_logging() -> None:
    """
    Configures the root logger based on the centralized logging configuration.
    This should be called exactly once at application startup.
    
    It validates that the logs directory exists and initializes the production
    logging format mapped to stdout and file storage.
    
    Using force=True ensures that any existing handlers are cleared,
    preventing duplicate logs during Streamlit reruns.
    """
    if RUNTIME.enable_logging:
        PATHS.logs_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, LOGGING.log_level),
            format=LOGGING.log_format,
            datefmt=LOGGING.date_format,
            force=True,  # Clears existing handlers to prevent duplicates
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(PATHS.logs_dir / "application.log")
            ]
        )
        
        logger = logging.getLogger("ai_pose_pro.config")
        logger.info("Logging initialized successfully.")
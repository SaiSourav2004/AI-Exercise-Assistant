import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from typing import Any, Optional, Callable
import config


def render_header() -> None:
    """Renders application header in the main container."""
    st.write("---------------------------------------------------------")
    st.title("AI-Based Exercise Pose Detection System")
    st.write("---------------------------------------------------------")


def format_label(label: Optional[str]) -> Optional[str]:
    """Converts raw ML labels to UI-friendly Title Case."""
    if not label:
        return None
    return label.replace("_", " ").title()


# ==========================================================
# SIDEBAR SECTIONS
# ==========================================================

def render_sidebar_header() -> None:
    """Renders the frozen left sidebar header and spacing."""

    st.sidebar.markdown("""
        <style>
            [data-testid="stSidebar"] .block-container {
                padding-top: 1.5rem !important;
            }

            [data-testid="stSidebar"] .stRadio [role="radiogroup"] {
                gap: 0.1rem !important;
            }

            [data-testid="stSidebar"] .stRadio {
                margin-top: -10px !important;
            }

            [data-testid="stSidebar"] [data-testid="stExpander"] details {
                margin-bottom: 0.1rem !important;
            }

            [data-testid="stSidebar"] hr {
                margin-top: 0.8rem !important;
                margin-bottom: 0.8rem !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("## 🧘 AI Fitness Assistant")
    st.sidebar.caption("Real-Time Exercise Pose Classification")
    st.sidebar.markdown("---")


def render_navigation() -> str:
    """Renders input-source navigation."""

    st.sidebar.write("### Navigation")

    choice = st.sidebar.radio(
        "Choose Input Source",
        config.INPUT_SOURCES,
        index=2,
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    return choice


def render_model_information(detector: Any) -> None:
    """Displays AI model information."""

    with st.sidebar.expander("🤖 AI Model", expanded=False):

        st.write("**Model:** Artificial Neural Network")
        st.write("**Framework:** TensorFlow / Keras")
        st.write("**Pose Detection:** MediaPipe BlazePose")
        st.write("**Input Features:** 132")

        num_classes = (
            len(detector.label_encoder.classes_)
            if detector and hasattr(detector, "label_encoder")
            else "Unknown"
        )

        st.write(f"**Exercise Classes:** {num_classes}")

        threshold = getattr(
            config,
            "MP_MIN_DETECTION_CONFIDENCE",
            "Unknown"
        )

        st.write(f"**Confidence Threshold:** {threshold}")
        st.write("**Model Status:** Loaded")


def render_supported_classes(detector: Any) -> None:
    """Displays supported exercise classes."""

    with st.sidebar.expander(
        "🏋 Supported Exercise Classes",
        expanded=False
    ):

        if detector and hasattr(detector, "label_encoder"):

            for cls in detector.label_encoder.classes_:
                st.write(f"- {format_label(cls)}")

        else:
            st.write("Classes not available.")


def render_system_status() -> None:
    """Displays current system status."""

    with st.sidebar.expander(
        "📊 System Status",
        expanded=False
    ):

        st.write("**AI Model:** Loaded")
        st.write("**MediaPipe:** Ready")
        st.write("**Camera:** Supported")
        st.write("**Prediction Engine:** Ready")


def render_future_roadmap() -> None:
    """Displays future development roadmap."""

    with st.sidebar.expander(
        "🚀 Future Roadmap",
        expanded=False
    ):

        st.write("**Workout Intelligence**")
        st.write("- Rep Counting")
        st.write("- Exercise Form Analysis")

        st.write("**Pose Analysis**")
        st.write("- Pose Correction")
        st.write("- Real-Time Form Feedback")

        st.write("**Analytics**")
        st.write("- Workout Summary")
        st.write("- Performance Tracking")


def render_developer_information() -> None:
    """Displays developer information."""

    with st.sidebar.expander(
        "👨‍💻 Developer",
        expanded=False
    ):

        st.write("**Developer:** Sai Sourav Panigrahi")
        st.write("**Role:** AI/ML & Data Science")
        st.write("**Project:** AI Exercise Assistant")

        st.markdown(
            "**GitHub:** [SaiSourav2004](https://github.com/SaiSourav2004)"
        )

        st.markdown(
            "**LinkedIn:** [Sai Sourav Panigrahi](https://www.linkedin.com/in/saisourav-panigrahi/)"
        )


def render_sidebar(detector: Any) -> str:
    """Master controller for the left sidebar."""

    render_sidebar_header()

    choice = render_navigation()

    render_model_information(detector)
    render_supported_classes(detector)
    render_system_status()
    render_future_roadmap()
    render_developer_information()

    return choice


# ==========================================================
# RIGHT DASHBOARD SECTION
# ==========================================================

def render_prediction_panel(
    prediction: Optional[str]
) -> None:

    st.markdown("**🎯 Current Exercise**")

    if prediction:
        st.header(format_label(prediction))
    else:
        st.header("Waiting...")


def render_confidence_panel(
    confidence: Optional[float]
) -> None:

    st.markdown("**📈 Confidence**")

    conf_val = (
        float(confidence)
        if confidence is not None
        else 0.0
    )

    st.progress(conf_val)

    st.caption(
        f"{conf_val * 100:.1f}%"
        if confidence is not None
        else "--"
    )


def render_status_panel(app_state: str) -> None:

    if app_state == "active":

        st.markdown("🟢 **Active**")

    elif app_state == "not_active":

        st.markdown("🔴 **Not Active**")

    elif app_state == "waiting_camera":

        st.markdown("🟡 **Waiting for Camera**")


def render_quick_metrics_panel(
    fps: Optional[float] = None,
    latency: Optional[float] = None
) -> None:

    c1, c2 = st.columns(2)

    c1.metric(
        "FPS",
        f"{fps:.1f}" if fps is not None else "--"
    )

    c2.metric(
        "Latency",
        f"{latency:.0f} ms"
        if latency is not None
        else "--"
    )


def render_right_dashboard(
    prediction: Optional[str] = None,
    confidence: Optional[float] = None,
    app_state: str = "waiting_camera",
    fps: Optional[float] = None,
    latency: Optional[float] = None,
    is_active: Optional[bool] = None
) -> None:
    """
    Renders the single Version 7 monitoring card.

    Displays:
    - Current Exercise
    - Confidence
    - Active / Not Active
    - FPS
    - Latency
    """

    # Backward compatibility with existing app.py
    if is_active is True:
        app_state = "active"

    elif is_active is False:
        app_state = "not_active"

    # Alignment correction
    # Keeps dashboard card aligned with the live camera section.
    st.markdown(
        "### <span style='opacity: 0;'>-</span>",
        unsafe_allow_html=True
    )

    st.caption(
        "<span style='opacity: 0;'>-</span>",
        unsafe_allow_html=True
    )

    # Single dashboard card
    with st.container(border=True):

        render_prediction_panel(prediction)

        st.write("")

        render_confidence_panel(confidence)

        st.write("")

        render_status_panel(app_state)

        st.divider()

        render_quick_metrics_panel(
            fps,
            latency
        )


# ==========================================================
# MAIN UI COMPONENTS
# ==========================================================

def render_image_uploader() -> Any:

    st.write("### Upload Image")

    return st.file_uploader(
        "Choose File",
        type=config.SUPPORTED_IMAGE_FORMATS,
        label_visibility="collapsed"
    )


def render_camera_input() -> Any:

    st.write("### Capture From Camera")

    return st.camera_input(
        "Capture Image",
        label_visibility="collapsed"
    )


def render_webrtc_player(
    video_frame_callback: Callable
) -> Any:
    """Renders the WebRTC live exercise camera."""

    st.write("### AI Live Monitoring")

    st.caption(
        "Real-Time Exercise Recognition"
    )

    rtc_config = RTCConfiguration(
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

    return webrtc_streamer(
        key="live-pose-detection",
        video_frame_callback=video_frame_callback,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )


def render_images(
    original_frame: Any,
    processed_frame: Any
) -> None:

    st.write(
        "### Processed Image (Pose Skeleton)"
    )

    st.image(
        processed_frame,
        use_column_width=True
    )

    with st.expander(
        "View Original Captured Image"
    ):

        st.image(
            original_frame,
            use_column_width=True
        )


def render_error(message: str) -> None:

    st.error(
        f"Error: {message}"
    )


def render_warning(message: str) -> None:

    st.warning(message)
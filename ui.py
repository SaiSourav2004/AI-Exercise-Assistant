"""
User Interface Module for AI Pose Pro.

Responsibilities:
    - Provides a premium, enterprise-grade presentation layer using Streamlit.
    - Manages global CSS injection, layout structures, and visual themes.
    - Renders the Header, Sidebar, Dashboard, Prediction Cards, and Footer.
    - Operates completely isolated from AI, camera, and application orchestration logic.

Data Contract:
    - Receives ONLY the immutable `PredictionOutput` dataclass for real-time updates.
    - Imports static visual constants and strings from `config.py`.

This is the designated FINAL FREEZE VERSION of the User Interface layer.
"""

import base64
from datetime import datetime
from typing import Optional, Tuple, Any, Dict

import streamlit as st

# Strictly importing from approved modules
from config import (
    APP,
    PATHS,
    MODELS,
    THEME,
    STATUS,
    UI
)
from mediapipe_detector import PredictionOutput


# ==============================================================================
# CSS INJECTION
# ==============================================================================

@st.cache_data
def _generate_global_css() -> str:
    """
    Generates a centralized, dynamic CSS block to style the Streamlit application.
    Cached to prevent unnecessary string formatting overhead on every rerun.
    
    Returns:
        str: The complete HTML style tag containing all application CSS.
    """
    return f"""
    <style>
        /* Typography Import */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');

        /* Global Application Reset */
        .stApp {{
            background-color: {THEME.colors.primary_bg};
            font-family: {THEME.typography.font_family};
            color: {THEME.colors.text_main};
        }}
        
        /* Hide default Streamlit decorations for enterprise look */
        header {{visibility: hidden;}}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        /* Scrollbar Styling */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: transparent;
        }}
        ::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(255, 255, 255, 0.2);
        }}

        /* Typography Classes */
        .ui-title {{
            font-size: {THEME.typography.size_title};
            font-weight: 700;
            color: {THEME.colors.text_main};
            margin-bottom: {THEME.layout.spacing_small};
            letter-spacing: -0.02em;
        }}
        .ui-header {{
            font-size: {THEME.typography.size_header};
            font-weight: 600;
            color: {THEME.colors.text_main};
            margin-bottom: {THEME.layout.spacing_medium};
            letter-spacing: -0.01em;
        }}
        .ui-label {{
            font-size: {THEME.typography.size_label};
            font-weight: 500;
            color: {THEME.colors.text_muted};
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .ui-metric-value {{
            font-size: {THEME.typography.size_metric};
            font-weight: 700;
            color: {THEME.colors.text_main};
            line-height: 1.1;
        }}
        .ui-small {{
            font-size: {THEME.typography.size_small};
            font-weight: 400;
            color: {THEME.colors.text_muted};
        }}

        /* Glassmorphism Card System */
        .glass-card {{
            background: {THEME.animation.glass_bg_rgba};
            backdrop-filter: blur({THEME.animation.glass_blur});
            -webkit-backdrop-filter: blur({THEME.animation.glass_blur});
            border: {THEME.animation.glass_border};
            border-radius: {THEME.layout.border_radius_large};
            padding: {THEME.layout.padding_large};
            box-shadow: {THEME.animation.shadow_medium};
            transition: {THEME.css_transition};
            margin-bottom: {THEME.layout.spacing_medium};
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
        }}
        .glass-card:hover {{
            box-shadow: {THEME.animation.shadow_large};
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.15);
        }}

        /* Badges */
        .badge {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            white-space: nowrap;
        }}
        .badge-success {{ background: rgba(16, 185, 129, 0.15); color: {THEME.colors.success}; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-warning {{ background: rgba(245, 158, 11, 0.15); color: {THEME.colors.warning}; border: 1px solid rgba(245, 158, 11, 0.3); }}
        .badge-error {{ background: rgba(239, 68, 68, 0.15); color: {THEME.colors.error}; border: 1px solid rgba(239, 68, 68, 0.3); }}
        .badge-accent {{ background: rgba(59, 130, 246, 0.15); color: {THEME.colors.accent}; border: 1px solid rgba(59, 130, 246, 0.3); }}
        .badge-neutral {{ background: rgba(255, 255, 255, 0.1); color: {THEME.colors.text_main}; border: 1px solid rgba(255, 255, 255, 0.2); }}

        /* Text Colors */
        .text-success {{ color: {THEME.colors.success} !important; }}
        .text-warning {{ color: {THEME.colors.warning} !important; }}
        .text-error {{ color: {THEME.colors.error} !important; }}
        .text-accent {{ color: {THEME.colors.accent} !important; }}
        
        /* Progress Bar */
        .progress-track {{
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 8px;
        }}
        .progress-fill {{
            height: 100%;
            background: {THEME.colors.accent};
            border-radius: 3px;
            transition: width 0.3s ease-in-out;
        }}

        /* Animations */
        @keyframes pulse {{
            0% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.7; transform: scale(0.98); }}
            100% {{ opacity: 1; transform: scale(1); }}
        }}
        .animate-pulse {{
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }}

        /* Sidebar Styling Overrides */
        [data-testid="stSidebar"] {{
            background-color: {THEME.colors.secondary_bg} !important;
            border-right: {THEME.animation.glass_border};
        }}
        
        /* Layout Adjustments */
        .dashboard-container {{
            display: flex;
            flex-direction: column;
            gap: {THEME.layout.spacing_medium};
        }}
        
        /* Responsive Utilities */
        @media (max-width: 768px) {{
            .ui-title {{ font-size: 1.5rem; }}
            .ui-metric-value {{ font-size: 2rem; }}
            .glass-card {{ padding: {THEME.layout.padding_medium}; }}
        }}
    </style>
    """


def inject_global_css() -> None:
    """Injects the globally cached CSS block into the Streamlit application."""
    st.markdown(_generate_global_css(), unsafe_allow_html=True)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _get_status_mapping(status: str) -> Dict[str, str]:
    """
    Centralized mapping dictionary for system and prediction statuses.
    
    Args:
        status: The string status from STATUS or PredictionOutput.
        
    Returns:
        Dict[str, str]: A dictionary containing 'color_class' and 'badge_class'.
    """
    mapping = {
        # Success States
        STATUS.sys_ready: {"color": "text-success", "badge": "badge-success"},
        STATUS.camera_ready: {"color": "text-success", "badge": "badge-success"},
        STATUS.model_loaded: {"color": "text-success", "badge": "badge-success"},
        STATUS.prediction_running: {"color": "text-success", "badge": "badge-success"},
        
        # Warning States
        STATUS.sys_warning: {"color": "text-warning", "badge": "badge-warning"},
        STATUS.det_lost: {"color": "text-warning", "badge": "badge-warning"},
        STATUS.det_no_person: {"color": "text-warning", "badge": "badge-warning"},
        STATUS.pred_calculating: {"color": "text-warning", "badge": "badge-warning"},
        STATUS.pred_unknown: {"color": "text-warning", "badge": "badge-warning"},
        
        # Error States
        STATUS.sys_error: {"color": "text-error", "badge": "badge-error"},
        STATUS.camera_error: {"color": "text-error", "badge": "badge-error"},
    }
    
    # Default to Accent for initial/transitional states (e.g., Initializing)
    return mapping.get(status, {"color": "text-accent", "badge": "badge-accent"})


@st.cache_data
def _encode_image_to_base64(image_path: Any) -> str:
    """
    Safely encodes an image to a base64 string for embedding within HTML.
    Cached to avoid repetitive I/O operations on UI refresh.
    
    Args:
        image_path: The pathlib.Path to the image asset.
        
    Returns:
        str: The base64 encoded string, or an empty string if reading fails.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception:
        return ""


# ==============================================================================
# COMPONENT RENDERERS
# ==============================================================================

def render_header() -> None:
    """
    Renders the global application header containing the logo, title, and model info.
    Designed for a premium, uncluttered presentation resembling a SaaS dashboard.
    """
    col1, col2 = st.columns([1, 11])
    logo_base64 = _encode_image_to_base64(PATHS.logo_path)
    
    with col1:
        if logo_base64:
            st.markdown(
                f'<img src="data:image/png;base64,{logo_base64}" width="60" style="border-radius: {THEME.layout.border_radius_small}; box-shadow: {THEME.animation.shadow_light};">',
                unsafe_allow_html=True
            )
        else:
            st.markdown(f'<div style="font-size: 2.5rem; text-shadow: {THEME.animation.shadow_light};">{UI.page_icon}</div>', unsafe_allow_html=True)
            
    with col2:
        st.markdown(f"""
            <div style="display: flex; flex-direction: column; justify-content: center; height: 100%;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span class="ui-title" style="margin-bottom: 0;">{UI.app_title}</span>
                    <span class="badge badge-accent">v{APP.full_version}</span>
                    <span class="badge badge-success">LIVE</span>
                </div>
                <span class="ui-small text-muted" style="margin-top: 4px;">Engine: <strong>{MODELS.framework.value}</strong> • Ready for inference</span>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown(f"<div style='margin-bottom: {THEME.layout.spacing_large};'></div>", unsafe_allow_html=True)


def render_sidebar() -> str:
    """
    Renders the navigation sidebar and persistent system information compact cards.
    
    Returns:
        str: The currently selected navigation menu item.
    """
    with st.sidebar:
        st.markdown(f"<div class='ui-header'>{UI.sidebar_title}</div>", unsafe_allow_html=True)
        
        # Navigation
        selected_page = st.radio(
            label="Menu",
            options=[UI.dashboard_title, UI.about_title],
            label_visibility="collapsed"
        )
        
        st.markdown(f"<hr style='border-color: rgba(255,255,255,0.1); margin: {THEME.layout.spacing_large} 0;'>", unsafe_allow_html=True)
        
        # System Information Compact Card
        st.markdown(f"""
            <div class="glass-card" style="padding: {THEME.layout.padding_medium}; margin-bottom: {THEME.layout.spacing_medium};">
                <div class="ui-label" style="margin-bottom: 8px;">System Profile</div>
                <div class="ui-small" style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span class="text-muted">Model:</span> <strong>{MODELS.model_type.name.replace('_', ' ')}</strong>
                </div>
                <div class="ui-small" style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span class="text-muted">Version:</span> <strong>{MODELS.version}</strong>
                </div>
                <div class="ui-small" style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span class="text-muted">Framework:</span> <strong>{MODELS.framework.value}</strong>
                </div>
                <div class="ui-small" style="display: flex; justify-content: space-between;">
                    <span class="text-muted">Classes:</span> <strong>{MODELS.output_class_count} Nodes</strong>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Tech Stack Compact Card
        stack_html = "".join([f"<span class='badge badge-neutral' style='margin: 0 4px 4px 0;'>{tech}</span>" for tech in APP.technology_stack])
        st.markdown(f"""
            <div class="glass-card" style="padding: {THEME.layout.padding_medium};">
                <div class="ui-label" style="margin-bottom: 12px;">Tech Stack</div>
                <div style="display: flex; flex-wrap: wrap;">
                    {stack_html}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        return str(selected_page)


def render_prediction_card(prediction: PredictionOutput) -> None:
    """
    Renders the primary prediction dashboard hero card.
    
    Args:
        prediction: The current frame's PredictionOutput contract.
    """
    conf_pct = int(prediction.confidence * 100)
    
    try:
        timestamp_str = datetime.fromtimestamp(prediction.timestamp).strftime('%H:%M:%S.%f')[:-3]
    except Exception:
        timestamp_str = "00:00:00.000"
    
    st.markdown(f"""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: {THEME.layout.spacing_medium};">
                <div class="ui-label">Live Classification</div>
                <div class="ui-small text-muted">{timestamp_str}</div>
            </div>
            
            <div class="ui-metric-value" style="margin-bottom: {THEME.layout.spacing_medium}; font-size: 3rem;">
                {prediction.predicted_label}
            </div>
            
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
                <span class="ui-label">Confidence Level</span>
                <div>
                    <span class="ui-metric-value text-accent" style="font-size: 1.5rem;">{conf_pct}%</span>
                    <span class="ui-small text-muted">({prediction.confidence:.4f})</span>
                </div>
            </div>
            
            <div class="progress-track">
                <div class="progress-fill" style="width: {conf_pct}%;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_metric_cards(prediction: PredictionOutput) -> None:
    """
    Renders secondary performance metrics inside premium glass cards.
    
    Args:
        prediction: The current frame's PredictionOutput contract.
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class="glass-card" style="padding: {THEME.layout.padding_medium};">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 1.2rem;">⏱️</span>
                    <span class="ui-label">Processing FPS</span>
                </div>
                <div class="ui-header" style="margin-bottom: 0;">{prediction.fps:.1f}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div class="glass-card" style="padding: {THEME.layout.padding_medium};">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 1.2rem;">🧠</span>
                    <span class="ui-label">AI Engine</span>
                </div>
                <div class="ui-header" style="margin-bottom: 0; font-size: 1.25rem;">{prediction.model_name}</div>
            </div>
        """, unsafe_allow_html=True)


def render_status_card(prediction: PredictionOutput) -> None:
    """
    Renders the health and status tracking indicator card using premium badges.
    
    Args:
        prediction: The current frame's PredictionOutput contract.
    """
    status_label = prediction.prediction_status
    status_styles = _get_status_mapping(status_label)
    
    detect_status = "ACTIVE" if prediction.is_person_detected else "LOST / IDLE"
    detect_badge = "badge-success" if prediction.is_person_detected else "badge-warning"
    
    st.markdown(f"""
        <div class="glass-card">
            <div class="ui-label" style="margin-bottom: {THEME.layout.spacing_medium};">{UI.health_title}</div>
            
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span class="ui-small text-muted">System State</span>
                <span class="badge {status_styles['badge']}">{status_label}</span>
            </div>
            
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="ui-small text-muted">Tracking State</span>
                <span class="badge {detect_badge}">{detect_status}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_dashboard(prediction: Optional[PredictionOutput]) -> None:
    """
    Orchestrates the entire right-side dashboard panel by aggregating all metric cards.
    Handles the empty/loading state gracefully with a premium pulse animation.
    
    Args:
        prediction: The current frame's PredictionOutput contract, or None.
    """
    st.markdown(f"<div class='ui-header'>{UI.dashboard_title}</div>", unsafe_allow_html=True)
    
    if not prediction:
        # Premium Loading State
        st.markdown(f"""
            <div class="glass-card animate-pulse" style="align-items: center; justify-content: center; min-height: 400px; text-align: center;">
                <div style="font-size: 3rem; margin-bottom: {THEME.layout.spacing_medium};">🎥</div>
                <div class="ui-title">Initializing Stream</div>
                <div class="ui-small text-muted" style="margin-top: {THEME.layout.spacing_small};">
                    • Waiting for Camera <br>
                    • Loading AI Engine <br>
                    • Preparing Layout
                </div>
            </div>
        """, unsafe_allow_html=True)
        return
        
    st.markdown("<div class='dashboard-container'>", unsafe_allow_html=True)
    render_prediction_card(prediction)
    render_metric_cards(prediction)
    render_status_card(prediction)
    st.markdown("</div>", unsafe_allow_html=True)


def create_main_layout() -> Tuple[Any, Any]:
    """
    Creates the main application layout grid according to the UDS constraints.
    Returns the Streamlit columns so the Application Layer (app.py) can inject the 
    WebRTC camera feed into the left panel and dashboard into the right panel.
    
    Returns:
        Tuple[Any, Any]: (Camera Column, Dashboard Column)
    """
    return st.columns([THEME.layout.camera_ratio, THEME.layout.dashboard_ratio])


def render_about_page() -> None:
    """
    Renders the static informational 'About' page using premium glass cards.
    """
    st.markdown(f"<div class='ui-title'>{UI.about_title}</div>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="glass-card">
            <div class="ui-header">{APP.name}</div>
            <p class="ui-small" style="line-height: 1.6; color: {THEME.colors.text_main}; font-size: 1rem;">
                {APP.description}
            </p>
            <p class="ui-small text-muted" style="line-height: 1.6;">
                Designed to operate completely locally with strict separation of concerns, 
                leveraging a robust Artificial Neural Network backed by MediaPipe's high-speed 
                pose estimation pipelines. The system is built for low-latency, real-time edge AI inference.
            </p>
            
            <hr style='border-color: rgba(255,255,255,0.1); margin: {THEME.layout.spacing_medium} 0;'>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
                <div>
                    <div class="ui-label" style="margin-bottom: 4px;">AI Model</div>
                    <div class="ui-small text-main"><strong>{MODELS.model_type.value}</strong></div>
                </div>
                <div>
                    <div class="ui-label" style="margin-bottom: 4px;">Framework</div>
                    <div class="ui-small text-main"><strong>{MODELS.framework.value}</strong></div>
                </div>
                <div>
                    <div class="ui-label" style="margin-bottom: 4px;">Developer</div>
                    <div class="ui-small text-main"><strong>{APP.author}</strong></div>
                </div>
                <div>
                    <div class="ui-label" style="margin-bottom: 4px;">License & Version</div>
                    <div class="ui-small text-main"><strong>{APP.license} • v{APP.full_version}</strong></div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_footer() -> None:
    """
    Renders the fixed application footer with dynamic copyright year and minimal metadata.
    """
    current_year = datetime.now().year
    
    st.markdown(f"""
        <div style="
            text-align: center; 
            margin-top: {THEME.layout.spacing_large}; 
            padding-top: {THEME.layout.spacing_medium};
            border-top: {THEME.animation.glass_border};
            color: {THEME.colors.text_muted};
        ">
            <div class="ui-small">
                © {current_year} <strong>{APP.name} v{APP.full_version}</strong> • Developed by {APP.author}
            </div>
            <div class="ui-small" style="margin-top: 4px; font-size: 0.75rem; opacity: 0.7;">
                Enterprise Computer Vision Architecture • Strictly Local Inference
            </div>
        </div>
    """, unsafe_allow_html=True)
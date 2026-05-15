import threading

import av
import cv2 as cv
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from utils.pipeline import GestureRecognitionPipeline


GESTURE_DESCRIPTIONS = {
    "Open hand": "Palm open with all fingers extended.",
    "Close hand": "Closed fist with fingers curled inward.",
    "Pointer": "Index finger extended for pointing.",
    "OK": "Thumb and index finger forming an OK sign.",
    "Rock": "Index and little finger extended.",
    "Good luck": "Gesture class trained as Good luck.",
    "Dislike": "Thumb-down gesture.",
    "Like": "Thumb-up gesture.",
}

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


st.set_page_config(
    page_title="Hand Gesture Recognition",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading MediaPipe and TensorFlow model...")
def load_pipeline():
    return GestureRecognitionPipeline()


class GestureVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pipeline = load_pipeline()
        self.lock = threading.Lock()
        self.latest_message = "Waiting for webcam frames"
        self.latest_label = "No prediction"
        self.latest_confidence = 0.0
        self.latest_probabilities = ()

    def recv(self, frame):
        bgr_frame = frame.to_ndarray(format="bgr24")
        rgb_frame = cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB)

        try:
            result = self.pipeline.process(rgb_frame)
            annotated_rgb = result.image if result.image is not None else rgb_frame
            primary_prediction = result.primary_prediction

            with self.lock:
                self.latest_message = result.message
                if primary_prediction:
                    self.latest_label = primary_prediction.label
                    self.latest_confidence = primary_prediction.confidence
                    self.latest_probabilities = primary_prediction.probabilities
                else:
                    self.latest_label = result.message
                    self.latest_confidence = 0.0
                    self.latest_probabilities = ()

        except Exception as exc:
            annotated_rgb = rgb_frame
            with self.lock:
                self.latest_message = f"Frame processing error: {exc}"
                self.latest_label = "Processing error"
                self.latest_confidence = 0.0
                self.latest_probabilities = ()

        annotated_bgr = cv.cvtColor(annotated_rgb, cv.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")


def render_styles():
    st.markdown(
        """
        <style>
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1240px;
            }
            .metric-card {
                border: 1px solid #d9e2ec;
                border-radius: 8px;
                padding: 1rem;
                background: #ffffff;
                min-height: 112px;
            }
            .section-panel {
                border: 1px solid #d9e2ec;
                border-radius: 8px;
                padding: 1rem 1.1rem;
                background: #fbfcfe;
            }
            .gesture-chip {
                display: inline-block;
                margin: 0.18rem 0.28rem 0.18rem 0;
                padding: 0.32rem 0.55rem;
                border: 1px solid #c8d3df;
                border-radius: 999px;
                background: #ffffff;
                color: #102a43;
                font-size: 0.92rem;
            }
            footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.header("Controls")
        st.caption("Use a well-lit scene and keep one hand clearly visible.")
        confidence_floor = st.slider(
            "Confidence reference",
            min_value=0,
            max_value=100,
            value=70,
            step=5,
            help="Visual reference only. The model prediction itself is unchanged.",
        )
        prefer_front_camera = st.toggle(
            "Prefer front camera",
            value=False,
            help="Requests the browser's user-facing camera when available.",
        )

        st.divider()
        st.header("Model")
        st.write("Input: 21 hand landmarks")
        st.write("Features: 42 normalized values")
        st.write("Classifier: TensorFlow/Keras")
        st.write("Detector: MediaPipe Hands")

        st.divider()
        st.caption("No frames are stored by this app.")

    return confidence_floor, prefer_front_camera


def render_header():
    st.title("Hand Gesture Recognition")
    st.markdown(
        "A Streamlit application for browser-based hand gesture recognition using "
        "MediaPipe landmarks and the existing TensorFlow/Keras classifier."
    )


def render_live_webcam(confidence_floor, prefer_front_camera):
    st.subheader("Live Webcam")
    st.write(
        "Start the webcam stream and show a supported gesture. Landmarks, bounding "
        "boxes, predicted label, and confidence are drawn directly on the live video."
    )

    media_stream_constraints = {
        "video": {"width": {"ideal": 960}, "height": {"ideal": 540}},
        "audio": False,
    }
    if prefer_front_camera:
        media_stream_constraints["video"]["facingMode"] = "user"

    ctx = webrtc_streamer(
        key="gesture-recognition-live",
        video_processor_factory=GestureVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints=media_stream_constraints,
        async_processing=True,
    )

    status_col, confidence_col = st.columns([1, 1])
    with status_col:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Current Gesture", "Live overlay")
        if ctx.video_processor:
            with ctx.video_processor.lock:
                st.caption(ctx.video_processor.latest_message)
        else:
            st.caption("Webcam is not running.")
        st.markdown("</div>", unsafe_allow_html=True)

    with confidence_col:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        confidence_value = 0.0
        if ctx.video_processor:
            with ctx.video_processor.lock:
                confidence_value = ctx.video_processor.latest_confidence
        st.metric("Confidence", f"{confidence_value * 100:.1f}%")
        st.progress(min(max(confidence_value, 0.0), 1.0))
        if confidence_value and confidence_value * 100 < confidence_floor:
            st.caption("Below the sidebar reference threshold.")
        else:
            st.caption("Reference threshold is for display only.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_snapshot_mode():
    st.subheader("Snapshot Test")
    st.write(
        "Use this section for a single browser camera capture or uploaded image. "
        "It is useful for checking exact confidence scores and class probabilities."
    )

    input_col, output_col = st.columns([1, 1])
    with input_col:
        camera_image = st.camera_input("Capture a hand gesture")
        uploaded_image = st.file_uploader(
            "Or upload an image",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
        )

    image_source = camera_image or uploaded_image
    if image_source is None:
        with output_col:
            st.info("Capture or upload an image to run snapshot inference.")
        return

    file_bytes = np.asarray(bytearray(image_source.read()), dtype=np.uint8)
    bgr_image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    if bgr_image is None:
        with output_col:
            st.error("Could not decode the selected image.")
        return

    rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)

    try:
        result = load_pipeline().process(rgb_image)
    except Exception as exc:
        with output_col:
            st.error(f"Model inference failed: {exc}")
        return

    with output_col:
        st.image(result.image, caption="Annotated result", use_column_width=True)
        if not result.has_hand:
            st.warning(result.message)
            return

        primary_prediction = result.primary_prediction
        st.success(result.message)
        st.metric("Top Prediction", primary_prediction.label)
        st.metric("Confidence", f"{primary_prediction.confidence * 100:.2f}%")

        labels = load_pipeline().labels
        probabilities = primary_prediction.probabilities
        if probabilities:
            score_rows = [
                {"Gesture": label, "Confidence": probabilities[index]}
                for index, label in enumerate(labels)
                if index < len(probabilities)
            ]
            st.dataframe(score_rows, use_container_width=True, hide_index=True)


def render_supported_gestures():
    st.subheader("Supported Gestures")
    chips = "".join(
        f'<span class="gesture-chip">{gesture}</span>'
        for gesture in GESTURE_DESCRIPTIONS
    )
    st.markdown(chips, unsafe_allow_html=True)

    with st.expander("Gesture guide", expanded=False):
        for gesture, description in GESTURE_DESCRIPTIONS.items():
            st.write(f"**{gesture}:** {description}")


def render_architecture():
    st.subheader("Architecture")
    left, right = st.columns([1, 1])

    with left:
        st.markdown(
            """
            <div class="section-panel">
            <strong>Inference pipeline</strong><br>
            Browser frame -> RGB conversion -> MediaPipe Hands -> 21 landmarks ->
            wrist-relative normalization -> Keras classifier -> annotated frame
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="section-panel">
            <strong>Deployment target</strong><br>
            Streamlit Community Cloud or local Streamlit runtime with
            streamlit-webrtc for browser webcam support.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_footer():
    st.divider()
    st.caption(
        "Built with Streamlit, streamlit-webrtc, MediaPipe Hands, OpenCV, "
        "TensorFlow/Keras, and NumPy."
    )


def main():
    render_styles()
    confidence_floor, prefer_front_camera = render_sidebar()
    render_header()

    try:
        load_pipeline()
    except Exception as exc:
        st.error(
            "The model or MediaPipe pipeline could not be loaded. "
            f"Details: {exc}"
        )
        st.stop()

    live_tab, snapshot_tab, details_tab = st.tabs(
        ["Live Recognition", "Snapshot Inference", "Project Details"]
    )

    with live_tab:
        render_live_webcam(confidence_floor, prefer_front_camera)

    with snapshot_tab:
        render_snapshot_mode()

    with details_tab:
        render_supported_gestures()
        render_architecture()

    render_footer()


if __name__ == "__main__":
    main()

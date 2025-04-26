"""Object detection demo with MobileNet SSD.
This model and code are based on
https://github.com/robmarkcole/object-detection-app
"""

import logging
import time
import base64
import queue
from pathlib import Path
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_session_memo import st_session_memo
from streamlit_webrtc import (
    WebRtcMode,
    webrtc_streamer,
    __version__ as st_webrtc_version,
)
import aiortc

import requests

@st.cache_data
def get_alarm_audio():
    """Load alarm sound file"""
    alarm_file = HERE / "alarm.mp3"
    if alarm_file.exists():
        with open(alarm_file, "rb") as f:
            return f.read()
    return None

def download_file(url: str, local_path: Path, expected_size: int):
    """Download a file from a URL to a local path."""
    if not local_path.exists() or local_path.stat().st_size != expected_size:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)


MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.caffemodel"
PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
PROTOTXT_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.prototxt.txt"

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray


@st.cache_resource  # type: ignore
def generate_label_colors():
    return np.random.uniform(0, 255, size=(len(CLASSES), 3))


COLORS = generate_label_colors()

download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)


@st_session_memo
def get_model():
    return cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))


net = get_model()

score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)

# NOTE: The callback will be called in another thread,
#       so use a queue here for thread-safety to pass the data
#       from inside to outside the callback.
# TODO: A general-purpose shared state object may be more useful.
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")

    # Run inference
    blob = cv2.dnn.blobFromImage(
        image=cv2.resize(image, (300, 300)),
        scalefactor=0.007843,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5),
    )
    net.setInput(blob)
    output = net.forward()

    h, w = image.shape[:2]

    # Convert the output array into a structured form.
    output = output.squeeze()  # (1, 1, N, 7) -> (N, 7)
    output = output[output[:, 2] >= score_threshold]
    detections = [
        Detection(
            class_id=int(detection[1]),
            label=CLASSES[int(detection[1])],
            score=float(detection[2]),
            box=(detection[3:7] * np.array([w, h, w, h])),
        )
        for detection in output
    ]

    # Render bounding boxes and captions
    for detection in detections:
        caption = f"{detection.label}: {round(detection.score * 100, 2)}%"
        color = COLORS[detection.class_id]
        xmin, ymin, xmax, ymax = detection.box.astype("int")

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(
            image,
            caption,
            (xmin, ymin - 15 if ymin - 15 > 15 else ymin + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    result_queue.put(detections)

    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Add alarm settings UI
st.sidebar.header("Alarm Settings")
enable_alarm = st.sidebar.checkbox("Enable person detection alarm", value=False)
alarm_cooldown = st.sidebar.slider("Alarm cooldown (seconds)", 1, 30, 5)

# Get the alarm audio data
alarm_audio = get_alarm_audio()
if enable_alarm and alarm_audio is None:
    st.sidebar.error("Alarm sound file not found. Please add 'alarm.mp3' to the app directory.")

# Initialize session state for alarm
if 'last_alarm_time' not in st.session_state:
    st.session_state.last_alarm_time = 0
if 'alarm_triggered' not in st.session_state:
    st.session_state.alarm_triggered = False

# Create placeholders for alarm UI elements
alarm_status = st.empty()
alarm_player = st.empty()

if st.checkbox("Show the detected labels", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        # NOTE: The video transformation with object detection and
        # this loop displaying the result labels are running
        # in different threads asynchronously.
        # Then the rendered video frames and the labels displayed here
        # are not strictly synchronized.
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)
            if enable_alarm and alarm_audio is not None:
                current_time = time.time()
                person_detected = any(detection.label == "person" for detection in result)
                
                if person_detected:
                    # Show alert message
                    alarm_status.warning("⚠️ ALERT: Person detected!")
                    
                    # Play alarm sound if cooldown has passed
                    if not st.session_state.alarm_triggered and (current_time - st.session_state.last_alarm_time) > alarm_cooldown:
                        st.session_state.last_alarm_time = current_time
                        st.session_state.alarm_triggered = True
                        audio_b64 = base64.b64encode(alarm_audio).decode()
                        alarm_player.markdown(
    f'<audio autoplay><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>',
    unsafe_allow_html=True
)
                else:
                    # Clear alarm status when no person detected
                    alarm_status.empty()
                    if st.session_state.alarm_triggered:
                        alarm_player.empty()
                        st.session_state.alarm_triggered = False

st.markdown(
    "This demo uses a model and code from "
    "https://github.com/robmarkcole/object-detection-app. "
    "Many thanks to the project."
)

st.markdown(
    f"Streamlit version: {st.__version__}  \n"
    f"Streamlit-WebRTC version: {st_webrtc_version}  \n"
    f"aiortc version: {aiortc.__version__}  \n"
)

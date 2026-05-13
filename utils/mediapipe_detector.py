import importlib


def _load_mediapipe_hands():
    for module_name in (
        "mediapipe.solutions.hands",
        "mediapipe.python.solutions.hands",
    ):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

    raise ModuleNotFoundError(
        "Could not import MediaPipe Hands. Install a MediaPipe version that "
        "includes the solutions API, for example: pip install mediapipe"
    )


mp_hands = _load_mediapipe_hands()

from utils.config import (
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)


class HandDetector:
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ):
        self._hands = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, rgb_image):
        return self._hands.process(rgb_image)

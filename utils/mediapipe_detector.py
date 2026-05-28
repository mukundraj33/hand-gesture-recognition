import importlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mediapipe as mp
import numpy as np

from utils.config import (
    HAND_LANDMARKER_TASK_PATH,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)


def _load_mediapipe_solutions_hands():
    for module_name in (
        "mediapipe.solutions.hands",
        "mediapipe.python.solutions.hands",
    ):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

    return None


@dataclass(frozen=True)
class _TasksResultAdapter:
    multi_hand_landmarks: tuple
    multi_handedness: tuple


class HandDetector:
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        task_model_path=HAND_LANDMARKER_TASK_PATH,
    ):
        self._mode = "solutions"
        self._hands = None
        self._landmarker = None

        mp_hands = _load_mediapipe_solutions_hands()
        if mp_hands is not None:
            self._hands = mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return

        self._mode = "tasks"
        task_path = Path(task_model_path)
        if not task_path.exists():
            raise FileNotFoundError(
                "MediaPipe Tasks requires the hand landmarker asset at "
                f"{task_path}. Download hand_landmarker.task into assets/."
            )

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        running_mode = vision.RunningMode.IMAGE
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(task_path)),
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, rgb_image):
        if self._mode == "solutions":
            return self._hands.process(rgb_image)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_image),
        )
        result = self._landmarker.detect(image)
        return self._adapt_tasks_result(result)

    @staticmethod
    def _adapt_tasks_result(result):
        landmarks = tuple(
            SimpleNamespace(landmark=hand_landmarks)
            for hand_landmarks in result.hand_landmarks
        )

        handedness = []
        for hand_handedness in result.handedness:
            if hand_handedness:
                label = hand_handedness[0].category_name
                score = hand_handedness[0].score
            else:
                label = "Unknown"
                score = 0.0

            handedness.append(
                SimpleNamespace(
                    classification=(
                        SimpleNamespace(label=label, score=score),
                    )
                )
            )

        return _TasksResultAdapter(
            multi_hand_landmarks=landmarks,
            multi_handedness=tuple(handedness),
        )

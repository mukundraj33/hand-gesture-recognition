import copy
from dataclasses import dataclass

import numpy as np

from model import KeyPointClassifier
from utils.config import LABEL_PATH, MODEL_PATH
from utils.drawing import draw_bounding_rect, draw_info_text, draw_landmarks
from utils.labels import load_labels
from utils.mediapipe_detector import HandDetector
from utils.preprocessing import (
    calc_bounding_rect,
    calc_landmark_list,
    pre_process_landmark,
)


@dataclass(frozen=True)
class GesturePrediction:
    handedness: str
    label: str
    confidence: float
    probabilities: tuple

    @property
    def display_text(self):
        return f"{self.handedness}: {self.label} ({self.confidence:.1%})"


@dataclass(frozen=True)
class PipelineResult:
    image: np.ndarray
    predictions: tuple
    message: str

    @property
    def primary_prediction(self):
        return self.predictions[0] if self.predictions else None

    @property
    def has_hand(self):
        return bool(self.predictions)


class GestureRecognitionPipeline:
    def __init__(self, model_path=MODEL_PATH, label_path=LABEL_PATH):
        self.detector = HandDetector()
        self.classifier = KeyPointClassifier(model_path=model_path)
        self.labels = load_labels(label_path)

    def predict(self, image):
        result = self.process(image)
        return result.image, result.message

    def process(self, image):
        if image is None:
            return PipelineResult(None, (), "No image received")

        rgb_image = self._ensure_rgb_uint8(image)
        debug_image = copy.deepcopy(rgb_image)
        process_image = copy.deepcopy(rgb_image)
        process_image.flags.writeable = False
        results = self.detector.process(process_image)
        process_image.flags.writeable = True

        if not results.multi_hand_landmarks:
            return PipelineResult(debug_image, (), "No hand detected")

        predictions = []
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            hand_sign_id, confidence, probabilities = (
                self.classifier.predict_with_confidence(pre_processed_landmark_list)
            )
            label = self._label_for_id(hand_sign_id)
            handedness_label = handedness.classification[0].label
            prediction = GesturePrediction(
                handedness=handedness_label,
                label=label,
                confidence=confidence,
                probabilities=tuple(float(value) for value in probabilities),
            )
            predictions.append(prediction)

            debug_image = draw_bounding_rect(debug_image, brect)
            debug_image = draw_landmarks(debug_image, landmark_list)
            debug_image = draw_info_text(
                debug_image,
                brect,
                handedness,
                f"{label} {confidence:.0%}",
            )

        message = " | ".join(prediction.display_text for prediction in predictions)
        return PipelineResult(debug_image, tuple(predictions), message)

    def _label_for_id(self, hand_sign_id):
        if 0 <= hand_sign_id < len(self.labels):
            return self.labels[hand_sign_id]
        return f"Unknown gesture ({hand_sign_id})"

    @staticmethod
    def _ensure_rgb_uint8(image):
        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        if image.shape[-1] == 4:
            image = image[:, :, :3]

        return image

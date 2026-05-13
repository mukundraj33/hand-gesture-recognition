import copy

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


class GestureRecognitionPipeline:
    def __init__(self, model_path=MODEL_PATH, label_path=LABEL_PATH):
        self.detector = HandDetector()
        self.classifier = KeyPointClassifier(model_path=model_path)
        self.labels = load_labels(label_path)

    def predict(self, image):
        if image is None:
            return None, "No image received"

        rgb_image = self._ensure_rgb_uint8(image)
        debug_image = copy.deepcopy(rgb_image)
        process_image = copy.deepcopy(rgb_image)
        process_image.flags.writeable = False
        results = self.detector.process(process_image)
        process_image.flags.writeable = True

        if not results.multi_hand_landmarks:
            return debug_image, "No hand detected"

        predictions = []
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            hand_sign_id = self.classifier(pre_processed_landmark_list)
            label = self._label_for_id(hand_sign_id)
            handedness_label = handedness.classification[0].label
            predictions.append(f"{handedness_label}: {label}")

            debug_image = draw_bounding_rect(debug_image, brect)
            debug_image = draw_landmarks(debug_image, landmark_list)
            debug_image = draw_info_text(debug_image, brect, handedness, label)

        return debug_image, " | ".join(predictions)

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

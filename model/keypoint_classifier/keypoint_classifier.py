import numpy as np
import tensorflow as tf


class KeyPointClassifier:
    def __init__(self, model_path='model/keypoint_classifier/keypoint_classifier.keras'):
        self.model = tf.keras.models.load_model(model_path)

    def __call__(self, landmark_list):
        predictions = self.predict_proba(landmark_list)
        return int(np.argmax(predictions))

    def predict_proba(self, landmark_list):
        input_array = np.array([landmark_list], dtype=np.float32)
        predictions = self.model.predict(input_array, verbose=0)
        return predictions[0]

    def predict_with_confidence(self, landmark_list):
        probabilities = self.predict_proba(landmark_list)
        result_index = int(np.argmax(probabilities))
        confidence = float(probabilities[result_index])
        return result_index, confidence, probabilities


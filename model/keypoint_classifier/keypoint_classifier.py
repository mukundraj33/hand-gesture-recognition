import numpy as np
import tensorflow as tf

class KeyPointClassifier:
    def __init__(self, model_path='model/keypoint_classifier/keypoint_classifier.keras'):
        self.model = tf.keras.models.load_model(model_path)

    def __call__(self, landmark_list):
        # Ensure input is in the right shape (batch size, input length)
        input_array = np.array([landmark_list], dtype=np.float32)
        predictions = self.model.predict(input_array)
        result_index = np.argmax(predictions[0])
        return result_index


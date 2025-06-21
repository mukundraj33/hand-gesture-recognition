# Hand Gesture Recognition System

## Project Overview
This real-time hand gesture recognition system identifies 8 distinct hand gestures using MediaPipe for landmark detection and a custom deep learning model for classification. The model achieves **93% accuracy** on the test set, demonstrating robust performance in recognizing complex hand gestures.

## Recognized Gestures
| Gesture        | Description                                  |
|----------------|----------------------------------------------|
| 0 Open Hand    | Palm fully open with fingers extended        |
| 1 Close Hand     | Fingers curled into a fist                   |
| 2 Pointer        | Index finger extended, others curled         |
| 3 OK             | Thumb and index finger forming a circle      |
| 4 Rock           | Index and pinky fingers extended             |
| 5 Good Luck      | Index and middle finger curled       |
| 6 Dislike        | Thumb pointing downward                      |
| 7 Like           | Thumb pointing upward                        |

## Technologies Used
- **Python 3.9+**: Primary programming language
- **MediaPipe**: Hand landmark detection
- **OpenCV**: Video processing and camera operations
- **TensorFlow/Keras**: Deep learning model development
- **scikit-learn**: Data processing and evaluation

## Key Skills Demonstrated
- Deep learning model design and training
- Real-time computer vision pipeline implementation
- MediaPipe integration for hand tracking
- Data collection and preprocessing for gesture recognition
- Model optimization and evaluation


## Model Architecture
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),   # 42 input features
    tf.keras.layers.Dropout(0.2),        # Regularization
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(8, activation='softmax')  # 8 output classes
])

Performance Metrics:
Accuracy: 93%
Precision: 94%
Recall: 93%
F1-Score: 93%

Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.76      0.86       152
           1       0.96      0.86      0.91       317
           2       0.91      0.97      0.94       203
           3       0.88      1.00      0.94       176
           4       0.99      0.94      0.96       257
           5       0.95      0.96      0.96       284
           6       0.87      0.99      0.93       174
           7       0.90      1.00      0.95       148

    accuracy                           0.93      1711
   macro avg       0.93      0.93      0.93      1711
weighted avg       0.94      0.93      0.93      1711
```

## Usage Instructions

### Keyboard Controls
| Key       | Functionality                                 |
|-----------|-----------------------------------------------|
| **0-7**   | Select gesture for data collection:<br>0 = Open Hand, 1 = Close Hand,<br>2 = Pointer, 3 = OK,<br>4 = Rock, 5 = Good Luck,<br>6 = Dislike, 7 = Like |
| **k**     | Enter logging mode (data collection)          |
| **n**     | Return to normal recognition mode             |
| **ESC**   | Exit application                              |

### Data Collection Workflow
1. **Enter Logging Mode**  
   Press the `k` key to enter data collection mode
   
2. **Select Gesture Class**  
   Press a number key (0-7) corresponding to the gesture you want to record:
   - 0: Open Hand
   - 1: Close Hand
   - 2: Pointer
   - 3: OK
   - 4: Rock
   - 5: Good Luck
   - 6: Dislike
   - 7: Like

3. **Perform Gesture**  
   Show the selected hand gesture to the camera. The system will automatically record landmark data.

4. **Collect Multiple Samples**  
   For best results:
   - Perform each gesture from different angles
   - Vary hand positions within frame
   - Repeat 20-30 times per gesture
   - Switch between gestures using number keys

5. **Finish Collection**  
   Press `n` to exit logging mode when done

6. **Retrain Model**  
   Use the collected data to retrain the model: keypoint_classification.ipynb
   ```bash
## Deployment
- **Android**: Convert model to TensorFlow Lite:
  ```bash
    import tensorflow as tf
    model = tf.keras.models.load_model('model/keypoint_classifier.keras')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('model/keypoint_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
    
- **Web**: Convert to TensorFlow.js

## Future Improvements
- **3D Gesture Recognition**  
  Integrate depth cameras for spatial gesture analysis
- **Sign Language Sequences**  
  Implement LSTM networks for continuous gesture recognition
- **Gesture Trajectory Analysis**  
  Incorporate movement velocity and path recognition
- **Real-Time Translation**  
  Add sign language to text/speech conversion
- **Gesture Security**  
  Develop authentication systems using unique gesture patterns
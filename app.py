import cv2 as cv
import numpy as np
import mediapipe as mp

import csv
import copy
import argparse
import itertools
from collections import deque

from utils import CvFpsCalc
from model import KeyPointClassifier


def get_args():
    """
    Parse command line arguments for camera configuration and model parameters.
    Returns parsed arguments object.
    """
    parser = argparse.ArgumentParser()

    # Camera configuration arguments
    parser.add_argument("--device", type=int, default=0, 
                        help="Camera device index (default=0)")
    parser.add_argument("--width", type=int, default=960,
                        help="Capture frame width (default=960)")
    parser.add_argument("--height", type=int, default=540,
                        help="Capture frame height (default=540)")
    
    # MediaPipe Hands model parameters
    parser.add_argument('--use_static_image_mode', action='store_true',
                        help="Use static image mode for MediaPipe")
    parser.add_argument("--min_detection_confidence", type=float, default=0.7,
                        help="Minimum detection confidence threshold")
    parser.add_argument("--min_tracking_confidence", type=int, default=0.5,
                        help="Minimum tracking confidence threshold")

    return parser.parse_args()


def main():
    """Main function for hand gesture recognition pipeline."""
    # Parse command line arguments
    args = get_args()
    
    # Initialize camera with specified settings
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # Initialize MediaPipe Hands model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    # Initialize gesture classifier
    keypoint_classifier = KeyPointClassifier()
    
    # Load gesture labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', 
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    # Initialize utilities
    cvFpsCalc = CvFpsCalc(buffer_len=10)  # FPS calculator
    point_history = deque(maxlen=16)       # Tracking point history
    use_brect = True                       # Toggle bounding rectangle display
    mode = 0                               # Current operation mode

    while True:
        # Calculate and display FPS
        fps = cvFpsCalc.get()
        
        # Process keyboard input
        key = cv.waitKey(10)
        if key == 27:  # ESC key to exit
            break
        number, mode = select_mode(key, mode)

        # Read camera frame
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)  # Create copy for drawing

        # Convert image to RGB format for MediaPipe
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)  # Process frame with MediaPipe
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            # Process each detected hand
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                 results.multi_handedness):
                # Calculate bounding box and landmarks
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Preprocess landmarks for classification
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                # Log data if in recording mode
                logging_csv(number, mode, pre_processed_landmark_list)

                # Classify hand gesture
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                # Update tracking history
                if hand_sign_id == "not applicable":
                    point_history.append(landmark_list[8])  # Track index fingertip
                else:
                    point_history.append([0, 0])

                # Draw annotations
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                )
        else:
            point_history.append([0, 0])

        # Draw tracking history and info overlay
        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Display result
        cv.imshow('Hand Gesture Recognition', debug_image)

    # Cleanup
    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    """
    Process keyboard input to change operation mode or select gestures.
    Returns:
        number: Selected gesture ID (-1 if none selected)
        mode: Current operation mode
    """
    number = -1
    # Number keys 0-9
    if 48 <= key <= 57:  # 0~9
        number = key - 48
    # Mode selection keys
    if key == 110:  # 'n' - normal mode
        mode = 0
    if key == 107:  # 'k' - keypoint logging mode
        mode = 1
    return number, mode


def calc_bounding_rect(image, landmarks):
    """
    Calculate bounding rectangle around hand landmarks.
    Returns rectangle as [x_min, y_min, x_max, y_max]
    """
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    # Convert normalized landmarks to pixel coordinates
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, [[landmark_x, landmark_y]], axis=0)
    
    # Calculate bounding rectangle
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    """
    Convert normalized landmarks to pixel coordinates in image space.
    Returns list of [x, y] coordinates for each landmark.
    """
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    # Convert each landmark to image coordinates
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    
    return landmark_point


def pre_process_landmark(landmark_list):
    """
    Preprocess landmarks for classification:
    1. Convert to relative coordinates
    2. Flatten to 1D list
    3. Normalize values
    """
    temp_landmark_list = copy.deepcopy(landmark_list)
    
    # Convert to relative coordinates based on wrist position
    base_x, base_y = temp_landmark_list[0]
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y
    
    # Flatten to 1D list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    # Normalize values by maximum absolute value
    max_value = max(map(abs, temp_landmark_list))
    return [n / max_value for n in temp_landmark_list]


def logging_csv(number, mode, landmark_list):
    """
    Log landmark data to CSV file for training when in recording mode.
    """
    # Only log in mode 1 (keypoint logging) with valid number
    if mode == 1 and (0 <= number <= 9):
        with open('model/keypoint_classifier/keypoint.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])


def draw_landmarks(image, landmark_point):
    """Draw hand landmarks and connections on the image."""
    if len(landmark_point) == 0:
        return image

    # Define connections between landmarks (bones)
    connections = [
        # Thumb
        (2, 3), (3, 4),
        # Index finger
        (5, 6), (6, 7), (7, 8),
        # Middle finger
        (9, 10), (10, 11), (11, 12),
        # Ring finger
        (13, 14), (14, 15), (15, 16),
        # Little finger
        (17, 18), (18, 19), (19, 20),
        # Palm
        (0, 1), (1, 2), (2, 5), (5, 9), 
        (9, 13), (13, 17), (17, 0)
    ]
    
    # Draw connections (bones)
    for connection in connections:
        start = tuple(landmark_point[connection[0]])
        end = tuple(landmark_point[connection[1]])
        # Draw thick black line and thin white line for contrast
        cv.line(image, start, end, (0, 0, 0), 6)
        cv.line(image, start, end, (255, 255, 255), 2)
    
    # Draw landmarks (joints)
    for i, point in enumerate(landmark_point):
        color = (255, 255, 255)  # Default white
        radius = 5  # Default size
        
        # Special styling for fingertips
        if i in [4, 8, 12, 16, 20]:  # Fingertips
            radius = 8
            cv.circle(image, tuple(point), radius, (0, 0, 0), 1)  # Black outline
        
        cv.circle(image, tuple(point), radius, color, -1)
    
    return image


def draw_bounding_rect(use_brect, image, brect):
    """Draw bounding rectangle around hand if enabled."""
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    """Draw information text above bounding box."""
    # Draw background rectangle for text
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)
    
    # Compose and display info text
    info_text = f"{handedness.classification[0].label[0:]}: {hand_sign_text}"
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


def draw_point_history(image, point_history):
    """Draw trail of tracked point history (index finger path)."""
    for i, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            # Vary circle size based on position in history
            cv.circle(image, tuple(point), 1 + int(i / 2), (152, 251, 152), 2)
    return image


def draw_info(image, fps, mode, number):
    """Draw system information overlay in top-left corner."""
    # FPS display
    cv.putText(image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    
    # Mode information
    mode_string = "Logging Key Point" if mode == 1 else "Normal Mode"
    if mode == 1:
        cv.putText(image, f"MODE: {mode_string}", (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, f"NUM: {number}", (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
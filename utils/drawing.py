import cv2 as cv


def draw_landmarks(image, landmark_point):
    if len(landmark_point) == 0:
        return image

    connections = [
        (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
        (0, 1), (1, 2), (2, 5), (5, 9),
        (9, 13), (13, 17), (17, 0),
    ]

    for connection in connections:
        start = tuple(landmark_point[connection[0]])
        end = tuple(landmark_point[connection[1]])
        cv.line(image, start, end, (0, 0, 0), 6)
        cv.line(image, start, end, (255, 255, 255), 2)

    for i, point in enumerate(landmark_point):
        color = (255, 255, 255)
        radius = 5

        if i in [4, 8, 12, 16, 20]:
            radius = 8
            cv.circle(image, tuple(point), radius, (0, 0, 0), 1)

        cv.circle(image, tuple(point), radius, color, -1)

    return image


def draw_bounding_rect(image, brect):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    handedness_label = handedness.classification[0].label
    info_text = f"{handedness_label}: {hand_sign_text}"
    cv.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )
    return image

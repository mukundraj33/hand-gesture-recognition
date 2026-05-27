import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _to_drawable(image):
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    return pil_image, ImageDraw.Draw(pil_image)


def _to_array(pil_image):
    return np.asarray(pil_image, dtype=np.uint8)


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

    pil_image, draw = _to_drawable(image)

    for start_index, end_index in connections:
        start = tuple(landmark_point[start_index])
        end = tuple(landmark_point[end_index])
        draw.line((start, end), fill=(0, 0, 0), width=6)
        draw.line((start, end), fill=(255, 255, 255), width=2)

    for index, point in enumerate(landmark_point):
        x, y = point
        radius = 8 if index in [4, 8, 12, 16, 20] else 5
        box = (x - radius, y - radius, x + radius, y + radius)

        if index in [4, 8, 12, 16, 20]:
            draw.ellipse(box, outline=(0, 0, 0), width=1)

        draw.ellipse(box, fill=(255, 255, 255))

    return _to_array(pil_image)


def draw_bounding_rect(image, brect):
    pil_image, draw = _to_drawable(image)
    draw.rectangle((brect[0], brect[1], brect[2], brect[3]), outline=(0, 0, 0), width=1)
    return _to_array(pil_image)


def draw_info_text(image, brect, handedness, hand_sign_text):
    pil_image, draw = _to_drawable(image)
    top = max(0, brect[1] - 22)
    draw.rectangle((brect[0], top, brect[2], brect[1]), fill=(0, 0, 0))

    handedness_label = handedness.classification[0].label
    info_text = f"{handedness_label}: {hand_sign_text}"
    font = ImageFont.load_default()
    draw.text((brect[0] + 5, top + 4), info_text, fill=(255, 255, 255), font=font)
    return _to_array(pil_image)

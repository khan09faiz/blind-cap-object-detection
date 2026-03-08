"""
Left / center / right direction classification from a bounding box.
"""

import numpy as np


def get_direction(bbox: np.ndarray, frame_width: int) -> str:
    """Classify the horizontal position of a bounding box.

    Args:
        bbox: xyxy array of shape (4,).
        frame_width: Width of the video frame in pixels.

    Returns:
        ``"left"``, ``"ahead"``, or ``"right"``.
    """
    center_x = (bbox[0] + bbox[2]) / 2.0

    if center_x < frame_width / 3:
        return "left"
    if center_x > 2 * frame_width / 3:
        return "right"
    return "ahead"

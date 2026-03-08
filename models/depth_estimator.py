"""
MiDaS monocular depth estimation wrapper.
All inference runs on CPU.
"""

import cv2
import numpy as np
import torch
from core.logging_config import get_logger

logger = get_logger(__name__)


class DepthEstimator:
    """MiDaS-small depth estimator running on CPU."""

    def __init__(self, model_name: str = "MiDaS_small") -> None:
        logger.info(f"Loading MiDaS model: {model_name}")
        self._model = torch.hub.load(
            "intel-isl/MiDaS", model_name, trust_repo=True
        )
        self._model.eval()
        self._model.to("cpu")

        transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        self._transform = transforms.small_transform
        logger.info("MiDaS model loaded (CPU)")

    @torch.no_grad()
    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth from a BGR OpenCV frame.

        Returns a float32 depth map (H x W) at the original frame resolution.
        """
        h, w = frame.shape[:2]

        # MiDaS expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(rgb).to("cpu")

        prediction = self._model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        return prediction.cpu().numpy().astype(np.float32)


def estimate_object_distance(depth_map: np.ndarray, bbox: np.ndarray) -> float:
    """Estimate distance to an object from its bounding-box center in the depth map.

    Args:
        depth_map: 2D float32 array (H x W) from DepthEstimator.estimate().
        bbox: xyxy array of shape (4,).

    Returns:
        Estimated distance in metres, clamped to [0.1, 50.0].
    """
    cx = int((bbox[0] + bbox[2]) / 2)
    cy = int((bbox[1] + bbox[3]) / 2)

    # Clamp to image bounds
    cy = max(0, min(cy, depth_map.shape[0] - 1))
    cx = max(0, min(cx, depth_map.shape[1] - 1))

    depth_value = float(depth_map[cy, cx])
    distance = 1.0 / (depth_value + 1e-6)
    return max(0.1, min(distance, 50.0))

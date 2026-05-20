"""MLX-only face analysis bootstrap for human source and driving frames."""

from __future__ import annotations

import cv2
import numpy as np

from .mlx_landmark_model import MlxLandmarkModel


class MlxFaceAnalysisModel:
    """Return face landmarks in the same list-of-arrays contract as MediaPipe."""

    def __init__(self, **kwargs):
        self.predict_type = "mlx"
        self.max_num_faces = int(kwargs.get("max_num_faces", 1))
        self.refine_iters = int(kwargs.get("refine_iters", 1))
        self.min_face_size = float(kwargs.get("min_face_size", 24.0))
        self.landmark = MlxLandmarkModel(**kwargs)

    def _is_valid_landmark(self, lmk, width, height):
        if not isinstance(lmk, np.ndarray) or lmk.ndim != 2 or lmk.shape[1] != 2:
            return False
        if lmk.shape[0] < 5 or not np.isfinite(lmk).all():
            return False

        left, top = np.min(lmk, axis=0)
        right, bottom = np.max(lmk, axis=0)
        bbox_w = float(right - left)
        bbox_h = float(bottom - top)
        if bbox_w < self.min_face_size or bbox_h < self.min_face_size:
            return False

        inside_x = np.logical_and(lmk[:, 0] >= -0.25 * width, lmk[:, 0] <= 1.25 * width)
        inside_y = np.logical_and(lmk[:, 1] >= -0.25 * height, lmk[:, 1] <= 1.25 * height)
        return float(np.mean(np.logical_and(inside_x, inside_y))) >= 0.75

    def predict(self, *data):
        img_bgr = data[0]
        if img_bgr is None:
            return []

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        height, width = img_bgr.shape[:2]
        lmk = self.landmark.predict(img_rgb)

        for _ in range(max(0, self.refine_iters)):
            if not self._is_valid_landmark(lmk, width, height):
                return []
            lmk = self.landmark.predict(img_rgb, lmk)

        if not self._is_valid_landmark(lmk, width, height):
            return []
        return [lmk.astype(np.float32)]

    def __del__(self):
        if hasattr(self, "landmark"):
            del self.landmark

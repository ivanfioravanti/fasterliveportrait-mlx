"""MLX-only face analysis bootstrap for human source and driving frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .mlx_landmark_model import MlxLandmarkModel


class MlxFaceAnalysisModel:
    """Return face landmarks in the same list-of-arrays contract as MediaPipe."""

    _FACE_CASCADE_NAMES = (
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_frontalface_alt.xml",
        "haarcascade_frontalface_default.xml",
    )

    def __init__(self, **kwargs):
        self.predict_type = "mlx"
        self.max_num_faces = int(kwargs.get("max_num_faces", 1))
        self.refine_iters = int(kwargs.get("refine_iters", 1))
        self.min_face_size = float(kwargs.get("min_face_size", 24.0))
        self.max_detector_dim = int(kwargs.get("max_detector_dim", 960))
        self.cascade_scale_factor = float(kwargs.get("cascade_scale_factor", 1.05))
        self.cascade_min_neighbors = int(kwargs.get("cascade_min_neighbors", 3))
        self.cascade_output_scale = float(kwargs.get("cascade_output_scale", 0.95))
        self.enable_face_cascade = bool(kwargs.get("enable_face_cascade", True))
        self.landmark = MlxLandmarkModel(**kwargs)
        self.face_cascades = self._load_face_cascades() if self.enable_face_cascade else []

    @staticmethod
    def _seed_landmarks_from_bbox(bbox):
        x, y, w, h = [float(v) for v in bbox]
        return np.array(
            [
                [x + 0.35 * w, y + 0.42 * h],
                [x + 0.65 * w, y + 0.42 * h],
                [x + 0.50 * w, y + 0.58 * h],
                [x + 0.40 * w, y + 0.76 * h],
                [x + 0.60 * w, y + 0.76 * h],
            ],
            dtype=np.float32,
        )

    def _load_face_cascades(self):
        cascade_dir = Path(cv2.data.haarcascades)
        cascades = []
        for cascade_name in self._FACE_CASCADE_NAMES:
            cascade = cv2.CascadeClassifier(str(cascade_dir / cascade_name))
            if not cascade.empty():
                cascades.append(cascade)
        return cascades

    @staticmethod
    def _expand_landmarks_to_min_size(lmk, min_size):
        if min_size is None:
            return lmk
        left, top = np.min(lmk, axis=0)
        right, bottom = np.max(lmk, axis=0)
        size = float(max(right - left, bottom - top))
        if size <= 1e-6 or size >= min_size:
            return lmk
        center = np.mean(lmk, axis=0, keepdims=True)
        return (lmk - center) * (float(min_size) / size) + center

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

    def _detect_face_seeds(self, img_bgr):
        if not self.face_cascades:
            return []

        height, width = img_bgr.shape[:2]
        scale = min(1.0, self.max_detector_dim / max(height, width))
        if scale < 1.0:
            detector_img = cv2.resize(
                img_bgr,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            detector_img = img_bgr

        gray = cv2.cvtColor(detector_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_size = max(16, int(round(self.min_face_size * scale)))

        detections = []
        for cascade in self.face_cascades:
            rects = cascade.detectMultiScale(
                gray,
                scaleFactor=self.cascade_scale_factor,
                minNeighbors=self.cascade_min_neighbors,
                minSize=(min_size, min_size),
            )
            detections.extend(rects.tolist() if len(rects) else [])

        if not detections:
            return []

        detections.sort(key=lambda rect: rect[2] * rect[3], reverse=True)
        inv_scale = 1.0 / scale
        seeds = []
        for rect in detections[: self.max_num_faces]:
            x, y, w, h = [v * inv_scale for v in rect]
            if min(w, h) >= self.min_face_size:
                seeds.append((self._seed_landmarks_from_bbox((x, y, w, h)), max(w, h) * self.cascade_output_scale))
        return seeds

    def _refine_landmark(self, img_rgb, seed, width, height, min_output_size=None):
        lmk = seed
        for _ in range(max(1, self.refine_iters)):
            if not self._is_valid_landmark(lmk, width, height):
                return None
            lmk = self.landmark.predict(img_rgb, lmk)
        lmk = self._expand_landmarks_to_min_size(lmk, min_output_size)
        if not self._is_valid_landmark(lmk, width, height):
            return None
        return lmk.astype(np.float32)

    def predict(self, *data):
        img_bgr = data[0]
        if img_bgr is None:
            return []

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        height, width = img_bgr.shape[:2]
        faces = []
        for seed, min_output_size in self._detect_face_seeds(img_bgr):
            lmk = self._refine_landmark(img_rgb, seed, width, height, min_output_size)
            if lmk is not None:
                faces.append(lmk)
        if faces:
            return faces[: self.max_num_faces]

        lmk = self.landmark.predict(img_rgb)
        lmk = self._refine_landmark(img_rgb, lmk, width, height)
        return [] if lmk is None else [lmk]

    def __del__(self):
        if hasattr(self, "landmark"):
            del self.landmark

"""Animal source face analysis without XPose or PyTorch runtime deps."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .mlx_face_analysis_model import MlxFaceAnalysisModel


class MlxAnimalFaceAnalysisModel:
    """Return animal crop landmarks for the same list-of-arrays pipeline contract.

    A packaged OpenCV cat-face cascade is preferred for cat-like sources and
    emits the 9-point layout already supported by crop.py. If that cascade
    cannot lock on, the MLX landmark bootstrap provides a no-PyTorch fallback
    for broader animal inputs.
    """

    _CAT_CASCADE_NAMES = (
        "haarcascade_frontalcatface_extended.xml",
        "haarcascade_frontalcatface.xml",
    )

    def __init__(self, **kwargs):
        self.predict_type = "mlx"
        self.max_num_faces = int(kwargs.get("max_num_faces", 1))
        self.min_face_size = int(kwargs.get("min_face_size", 40))
        self.max_detector_dim = int(kwargs.get("max_detector_dim", 960))
        self.cascade_scale_factor = float(kwargs.get("cascade_scale_factor", 1.05))
        self.cascade_min_neighbors = int(kwargs.get("cascade_min_neighbors", 3))
        self.enable_mlx_bootstrap = bool(kwargs.get("enable_mlx_bootstrap", True))
        self.enable_cat_cascade = bool(kwargs.get("enable_cat_cascade", True))
        self.prefer_cat_cascade = bool(kwargs.get("prefer_cat_cascade", True))

        self.bootstrap = MlxFaceAnalysisModel(**kwargs) if self.enable_mlx_bootstrap else None
        self.cat_cascades = self._load_cat_cascades() if self.enable_cat_cascade else []

    @staticmethod
    def _landmarks_from_bbox(bbox):
        x, y, w, h = [float(v) for v in bbox]
        return np.array(
            [
                [x + 0.84 * w, y + 0.40 * h],  # right eye outer
                [x + 0.64 * w, y + 0.45 * h],  # right eye inner
                [x + 0.36 * w, y + 0.405 * h],  # left eye inner
                [x + 0.16 * w, y + 0.382 * h],  # left eye outer
                [x + 0.49 * w, y + 0.725 * h],  # nose
                [x + 0.335 * w, y + 0.79 * h],  # mouth right
                [x + 0.63 * w, y + 0.785 * h],  # mouth left
                [x + 0.50 * w, y + 0.75 * h],  # upper mouth
                [x + 0.50 * w, y + 0.80 * h],  # lower mouth
            ],
            dtype=np.float32,
        )

    def _load_cat_cascades(self):
        cascade_dir = Path(cv2.data.haarcascades)
        cascades = []
        for cascade_name in self._CAT_CASCADE_NAMES:
            cascade = cv2.CascadeClassifier(str(cascade_dir / cascade_name))
            if not cascade.empty():
                cascades.append(cascade)
        return cascades

    def _predict_cat_cascade(self, img_bgr):
        if not self.cat_cascades:
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
        for cascade in self.cat_cascades:
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
        faces = []
        inv_scale = 1.0 / scale
        for rect in detections[: self.max_num_faces]:
            x, y, w, h = [v * inv_scale for v in rect]
            if min(w, h) < self.min_face_size:
                continue
            faces.append(self._landmarks_from_bbox((x, y, w, h)))
        return faces

    def predict(self, *data):
        img_bgr = data[0]
        if img_bgr is None:
            return []

        if self.prefer_cat_cascade:
            faces = self._predict_cat_cascade(img_bgr)
            if faces:
                return faces[: self.max_num_faces]

        if self.bootstrap is not None:
            faces = self.bootstrap.predict(img_bgr)
            if faces:
                return faces[: self.max_num_faces]

        return self._predict_cat_cascade(img_bgr)

    def predict_dense(self, img_bgr):
        """Return dense MLX landmarks when callers need eye/lip ratios.

        The default animal crop path intentionally returns XPose-style 9-point
        landmarks for stable cat crops. Eye/lip retargeting needs the dense
        eyelid and mouth contour indices used by the human retargeting MLPs, so
        it asks the MLX bootstrap detector directly.
        """
        if img_bgr is None or self.bootstrap is None:
            return []
        faces = self.bootstrap.predict(img_bgr)
        return faces[: self.max_num_faces] if faces else []

    def __del__(self):
        if hasattr(self, "bootstrap"):
            del self.bootstrap

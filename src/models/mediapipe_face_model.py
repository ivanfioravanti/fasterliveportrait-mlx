# -*- coding: utf-8 -*-
# @Time    : 2024/8/7 9:00
# @Author  : shaoguowen
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: mediapipe_face_model.py
from contextlib import contextmanager
import os
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np

try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass


@contextmanager
def _suppress_native_stderr(enabled=True):
    if not enabled:
        yield
        return
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)


class MediaPipeFaceModel:
    """
    MediaPipeFaceModel
    """

    def __init__(self, **kwargs):
        self.backend = "solutions" if hasattr(mp, "solutions") else "tasks"
        self.quiet_init = kwargs.get("quiet_init", True)
        self.face_mesh = (
            self._init_solutions(kwargs)
            if self.backend == "solutions"
            else self._init_tasks(kwargs)
        )

    def _init_solutions(self, kwargs):
        mp_face_mesh = mp.solutions.face_mesh
        with _suppress_native_stderr(self.quiet_init):
            return mp_face_mesh.FaceMesh(
                static_image_mode=kwargs.get("static_image_mode", True),
                max_num_faces=kwargs.get("max_num_faces", 1),
                refine_landmarks=kwargs.get("refine_landmarks", True),
                min_detection_confidence=kwargs.get("det_thresh", kwargs.get("min_detection_confidence", 0.15)))

    def _init_tasks(self, kwargs):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        model_path = Path(kwargs.get("model_path", "./checkpoints/mediapipe/face_landmarker.task"))
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe task model not found: {model_path}. "
                "Download it with: curl -L https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/latest/face_landmarker.task "
                "-o checkpoints/mediapipe/face_landmarker.task"
            )

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=kwargs.get("max_num_faces", 1),
            min_face_detection_confidence=kwargs.get("det_thresh", kwargs.get("min_detection_confidence", 0.15)),
            min_face_presence_confidence=kwargs.get("min_face_presence_confidence", 0.15),
            min_tracking_confidence=kwargs.get("min_tracking_confidence", 0.15))
        with _suppress_native_stderr(self.quiet_init):
            return vision.FaceLandmarker.create_from_options(options)

    def predict(self, *data):
        img_bgr = data[0]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]
        if self.backend == "tasks":
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb))
            results = self.face_mesh.detect(image)
            multi_face_landmarks = results.face_landmarks
        else:
            results = self.face_mesh.process(img_rgb)
            multi_face_landmarks = results.multi_face_landmarks

        # Print and draw face mesh landmarks on the image.
        if not multi_face_landmarks:
            return []
        outs = []
        for face_landmarks in multi_face_landmarks:
            landmarks = []
            for landmark in getattr(face_landmarks, "landmark", face_landmarks):
                # 提取每个关键点的 x, y, z 坐标
                landmarks.append([landmark.x * w, landmark.y * h])
            outs.append(np.array(landmarks))
        return outs

"""Face detection wrapper.

Primary backend: Mediapipe Face Detection (`pip install mediapipe`).
Fallback backend: OpenCV Haar Cascade (always available with opencv-python*).

A unified `detect(bgr)` returns a list of dicts:
    {"bbox": [x, y, w, h], "score": float, "center": [cx, cy]}
sorted by score descending (mediapipe) or area descending (haar).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class _HaarBackend:
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: int = 60

    def __post_init__(self) -> None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"无法加载 Haar 级联: {path}")

    def detect(self, bgr: np.ndarray) -> list[dict[str, Any]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=float(self.scale_factor),
            minNeighbors=int(self.min_neighbors),
            minSize=(int(self.min_size), int(self.min_size)),
        )
        out: list[dict[str, Any]] = []
        for (x, y, w, h) in faces:
            out.append(
                {
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "score": 1.0,
                    "center": [int(x + w / 2), int(y + h / 2)],
                }
            )
        out.sort(key=lambda d: d["bbox"][2] * d["bbox"][3], reverse=True)
        return out


@dataclass
class _MediapipeBackend:
    min_confidence: float = 0.5
    model_selection: int = 0

    def __post_init__(self) -> None:
        import mediapipe as mp

        self._mp_fd = mp.solutions.face_detection.FaceDetection(
            model_selection=int(self.model_selection),
            min_detection_confidence=float(self.min_confidence),
        )

    def detect(self, bgr: np.ndarray) -> list[dict[str, Any]]:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._mp_fd.process(rgb)
        out: list[dict[str, Any]] = []
        if not res.detections:
            return out
        for det in res.detections:
            r = det.location_data.relative_bounding_box
            x = max(int(r.xmin * w), 0)
            y = max(int(r.ymin * h), 0)
            bw = min(int(r.width * w), w - x)
            bh = min(int(r.height * h), h - y)
            if bw <= 0 or bh <= 0:
                continue
            score = float(det.score[0]) if det.score else 0.0
            out.append(
                {
                    "bbox": [x, y, bw, bh],
                    "score": round(score, 3),
                    "center": [int(x + bw / 2), int(y + bh / 2)],
                }
            )
        out.sort(key=lambda d: d["score"], reverse=True)
        return out


class FaceDetector:
    """High-level face detector with backend selection and lazy init."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = dict(cfg or {})
        self.backend_name = str(cfg.get("backend", "haar")).lower()
        self._cfg = cfg
        self._impl: Any = None
        self._effective_backend: str = ""

    def _build(self) -> Any:
        if self.backend_name == "mediapipe":
            try:
                impl = _MediapipeBackend(
                    min_confidence=float(self._cfg.get("mediapipe_min_confidence", 0.5)),
                    model_selection=int(self._cfg.get("mediapipe_model_selection", 0)),
                )
                self._effective_backend = "mediapipe"
                return impl
            except ImportError:
                print("[face] mediapipe 未安装，自动回退到 haar；安装：pip install mediapipe")
        impl = _HaarBackend(
            scale_factor=float(self._cfg.get("haar_scale_factor", 1.1)),
            min_neighbors=int(self._cfg.get("haar_min_neighbors", 5)),
            min_size=int(self._cfg.get("haar_min_size", 60)),
        )
        self._effective_backend = "haar"
        return impl

    def detect(self, bgr: np.ndarray) -> list[dict[str, Any]]:
        if self._impl is None:
            self._impl = self._build()
        return self._impl.detect(bgr)

    @property
    def effective_backend(self) -> str:
        if self._impl is None:
            self._build()
        return self._effective_backend


def pick_face(
    faces: list[dict[str, Any]],
    img_w: int,
    img_h: int,
    *,
    strategy: str = "largest",
) -> dict[str, Any] | None:
    if not faces:
        return None
    if strategy == "center":
        cx0, cy0 = img_w / 2.0, img_h / 2.0
        return min(
            faces,
            key=lambda f: (f["center"][0] - cx0) ** 2 + (f["center"][1] - cy0) ** 2,
        )
    return max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])


def draw_faces(
    bgr: np.ndarray,
    faces: list[dict[str, Any]],
    *,
    primary: dict[str, Any] | None = None,
) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    cv2.line(out, (w // 2, 0), (w // 2, h), (60, 60, 60), 1)
    cv2.line(out, (0, h // 2), (w, h // 2), (60, 60, 60), 1)
    cv2.circle(out, (w // 2, h // 2), 4, (60, 60, 60), -1)

    for f in faces:
        x, y, bw, bh = f["bbox"]
        is_pri = primary is not None and f is primary
        color = (0, 220, 80) if is_pri else (200, 200, 200)
        thickness = 2 if is_pri else 1
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, thickness)
        cx, cy = f["center"]
        cv2.circle(out, (cx, cy), 4, color, -1)
        score = f.get("score", 0.0)
        cv2.putText(
            out,
            f"face {score:.2f}" if score < 1 else "face",
            (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    if primary is not None:
        cx, cy = primary["center"]
        cv2.line(out, (w // 2, h // 2), (cx, cy), (0, 220, 80), 1)
    return out

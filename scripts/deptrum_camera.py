"""Python wrapper for the Deptrum Aurora 930 depth camera via the C bridge.

Usage::

    from deptrum_camera import DeptrumCamera

    cam = DeptrumCamera()
    cam.open()
    cam.start()

    ok, bgr = cam.read()          # like cv2.VideoCapture.read()
    ok, depth = cam.read_depth()

    cam.release()
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import numpy as np

_LIB: Optional[ctypes.CDLL] = None
_SDK_DIR = Path(__file__).resolve().parents[1] / "sdk" / "deptrum"
_BRIDGE_SO = _SDK_DIR / "lib" / "libdeptrum_bridge.so"

# Maximum expected resolution (allocation ceilings)
_MAX_W, _MAX_H = 1920, 1080


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB

    if not _BRIDGE_SO.exists():
        raise FileNotFoundError(
            f"Deptrum bridge library not found at {_BRIDGE_SO}. "
            "Run: bash sdk/deptrum/build.sh"
        )

    _LIB = ctypes.CDLL(str(_BRIDGE_SO))

    # int deptrum_device_count()
    _LIB.deptrum_device_count.restype = ctypes.c_int
    _LIB.deptrum_device_count.argtypes = []

    # int deptrum_open(int res_idx, int rgb, int ir, int depth, int pc)
    _LIB.deptrum_open.restype = ctypes.c_int
    _LIB.deptrum_open.argtypes = [ctypes.c_int] * 5

    # int deptrum_start()
    _LIB.deptrum_start.restype = ctypes.c_int
    _LIB.deptrum_start.argtypes = []

    # int deptrum_grab(rgb, rw, rh, depth, dw, dh, ir, iw, ih, timeout)
    _LIB.deptrum_grab.restype = ctypes.c_int
    _LIB.deptrum_grab.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]

    # int deptrum_stop()
    _LIB.deptrum_stop.restype = ctypes.c_int
    _LIB.deptrum_stop.argtypes = []

    # int deptrum_close()
    _LIB.deptrum_close.restype = ctypes.c_int
    _LIB.deptrum_close.argtypes = []

    # const char* deptrum_sdk_version()
    _LIB.deptrum_sdk_version.restype = ctypes.c_char_p
    _LIB.deptrum_sdk_version.argtypes = []

    return _LIB


def device_count() -> int:
    """Return the number of Deptrum cameras connected (0 if none)."""
    try:
        lib = _load_lib()
        n = lib.deptrum_device_count()
        return max(n, 0)
    except Exception:
        return 0


def sdk_version() -> str:
    try:
        lib = _load_lib()
        v = lib.deptrum_sdk_version()
        return v.decode("utf-8", errors="replace") if v else "unknown"
    except Exception:
        return "unknown"


class DeptrumCamera:
    """OpenCV-style wrapper for the Deptrum Aurora 930 depth camera."""

    def __init__(
        self,
        resolution_mode_index: int = 2,
        rgb: bool = True,
        ir: bool = True,
        depth: bool = True,
        point_cloud: bool = False,
        timeout_ms: int = 2000,
    ):
        self._lib = _load_lib()
        self._res_idx = resolution_mode_index
        self._rgb = rgb
        self._ir = ir
        self._depth = depth
        self._pc = point_cloud
        self._timeout = timeout_ms

        self._opened = False
        self._streaming = False

        self._rgb_buf = np.zeros(_MAX_H * _MAX_W * 3, dtype=np.uint8)
        self._depth_buf = np.zeros(_MAX_H * _MAX_W, dtype=np.uint16)
        self._ir_buf = np.zeros(_MAX_H * _MAX_W, dtype=np.uint8)

    # ── lifecycle ────────────────────────────────────────────────

    def open(self) -> None:
        if self._opened:
            return
        st = self._lib.deptrum_open(
            self._res_idx,
            int(self._rgb), int(self._ir), int(self._depth), int(self._pc),
        )
        if st != 0:
            raise RuntimeError(f"deptrum_open failed (code {st})")
        self._opened = True

    def start(self) -> None:
        if not self._opened:
            self.open()
        if self._streaming:
            return
        st = self._lib.deptrum_start()
        if st != 0:
            raise RuntimeError(f"deptrum_start failed (code {st})")
        self._streaming = True

    def isOpened(self) -> bool:  # noqa: N802 – matches OpenCV API
        return self._opened and self._streaming

    def release(self) -> None:
        if self._streaming:
            self._lib.deptrum_stop()
            self._streaming = False
        if self._opened:
            self._lib.deptrum_close()
            self._opened = False

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    def __enter__(self):
        self.open()
        self.start()
        return self

    def __exit__(self, *exc):
        self.release()

    # ── frame access ─────────────────────────────────────────────

    def _grab(self, want_rgb: bool, want_depth: bool, want_ir: bool):
        """Low-level grab; returns raw buffers + dimensions."""
        if not self._streaming:
            self.start()

        rw, rh = ctypes.c_int(0), ctypes.c_int(0)
        dw, dh = ctypes.c_int(0), ctypes.c_int(0)
        iw, ih = ctypes.c_int(0), ctypes.c_int(0)

        rgb_ptr = self._rgb_buf.ctypes.data if want_rgb else None
        dep_ptr = self._depth_buf.ctypes.data if want_depth else None
        ir_ptr  = self._ir_buf.ctypes.data if want_ir else None

        st = self._lib.deptrum_grab(
            rgb_ptr, ctypes.byref(rw), ctypes.byref(rh),
            dep_ptr, ctypes.byref(dw), ctypes.byref(dh),
            ir_ptr,  ctypes.byref(iw), ctypes.byref(ih),
            self._timeout,
        )
        return st, (rw.value, rh.value), (dw.value, dh.value), (iw.value, ih.value)

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read one BGR frame. Returns (ok, frame) like cv2.VideoCapture.read()."""
        st, (w, h), _, _ = self._grab(True, False, False)
        if st != 0 or w == 0:
            return False, None

        if h < 0:
            # NV12 encoded, need OpenCV conversion
            import cv2
            abs_h = -h
            nv12 = self._rgb_buf[: int(w * abs_h * 1.5)].reshape(int(abs_h * 1.5), w)
            bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
            return True, bgr

        bgr = self._rgb_buf[: w * h * 3].reshape(h, w, 3).copy()
        return True, bgr

    def read_depth(self) -> tuple[bool, np.ndarray | None]:
        """Read one 16-bit depth frame (values in mm)."""
        st, _, (w, h), _ = self._grab(False, True, False)
        if st != 0 or w == 0 or h == 0:
            return False, None
        depth = self._depth_buf[: w * h].reshape(h, w).copy()
        return True, depth

    def read_ir(self) -> tuple[bool, np.ndarray | None]:
        """Read one 8-bit IR frame."""
        st, _, _, (w, h) = self._grab(False, False, True)
        if st != 0 or w == 0 or h == 0:
            return False, None
        ir = self._ir_buf[: w * h].reshape(h, w).copy()
        return True, ir

    def read_all(self):
        """Read RGB + depth + IR in one call. Returns (ok, bgr, depth, ir)."""
        st, (rw, rh), (dw, dh), (iw, ih) = self._grab(True, True, True)
        if st != 0:
            return False, None, None, None

        bgr = None
        if rw > 0:
            if rh < 0:
                import cv2
                abs_h = -rh
                nv12 = self._rgb_buf[:int(rw * abs_h * 1.5)].reshape(int(abs_h * 1.5), rw)
                bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
            else:
                bgr = self._rgb_buf[:rw * rh * 3].reshape(rh, rw, 3).copy()

        depth = None
        if dw > 0 and dh > 0:
            depth = self._depth_buf[:dw * dh].reshape(dh, dw).copy()

        ir = None
        if iw > 0 and ih > 0:
            ir = self._ir_buf[:iw * ih].reshape(ih, iw).copy()

        return True, bgr, depth, ir

    # ── OpenCV VideoCapture compatibility ────────────────────────

    def get(self, prop_id: int) -> float:
        """Minimal get() for CAP_PROP_FRAME_WIDTH / HEIGHT."""
        import cv2
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(_MAX_W)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(_MAX_H)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        return False

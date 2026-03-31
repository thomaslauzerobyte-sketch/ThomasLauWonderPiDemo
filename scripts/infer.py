#!/usr/bin/env python3
"""单帧或批量图像推理：内置 Haar 人脸检测演示；可选 ONNX 分类模型。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import load_config, resolve_path


def _load_class_names(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def detect_haar(gray: np.ndarray) -> list[dict[str, Any]]:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError("无法加载 Haar 级联文件（OpenCV 安装可能不完整）。")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    out: list[dict[str, Any]] = []
    for (x, y, w, h) in faces:
        out.append({"label": "face", "bbox": [int(x), int(y), int(w), int(h)], "score": 1.0})
    return out


def _onnx_preprocess(bgr: np.ndarray, size: tuple[int, int], normalize: bool) -> np.ndarray:
    h, w = size
    img = cv2.resize(bgr, (w, h))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    if normalize:
        rgb = rgb / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def classify_onnx(
    bgr: np.ndarray,
    model_path: Path,
    input_size: tuple[int, int],
    normalize: bool,
    class_names: list[str],
) -> list[dict[str, Any]]:
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError("未安装 onnxruntime，请: pip install onnxruntime（或 onnxruntime-lite）") from e

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError("ONNX 模型无输入节点")
    in_name = inputs[0].name
    x = _onnx_preprocess(bgr, input_size, normalize)
    out = session.run(None, {in_name: x})[0]
    logits = np.array(out).reshape(-1)
    if logits.size == 0:
        return []
    exp = np.exp(logits - np.max(logits))
    prob = exp / np.sum(exp)
    top_i = int(np.argmax(prob))
    score = float(prob[top_i])
    label = class_names[top_i] if top_i < len(class_names) else f"class_{top_i}"
    return [{"label": label, "score": score, "bbox": None}]


def infer_image(
    image_path: Path,
    method: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"无法读取图像: {image_path}")

    inf = cfg.get("inference", {})
    paths = cfg.get("paths", {})
    if method == "haar":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dets = detect_haar(gray)
    elif method == "onnx":
        model_path = resolve_path(paths.get("model_path", "models/model.onnx"))
        if not model_path.is_file():
            raise FileNotFoundError(f"未找到 ONNX 模型: {model_path}")
        sz = inf.get("onnx_input_size", [224, 224])
        h, w = int(sz[0]), int(sz[1])
        names_path = paths.get("class_names")
        names = _load_class_names(resolve_path(names_path) if names_path else None)
        dets = classify_onnx(
            bgr,
            model_path,
            (h, w),
            bool(inf.get("normalize", True)),
            names,
        )
    else:
        raise ValueError(f"未知 inference.method: {method}")

    return {"image": str(image_path.resolve()), "detections": dets}


def _list_images(d: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="WonderPi Demo：图像推理")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--source", type=Path, required=True, help="图像文件或目录")
    parser.add_argument("--method", choices=("haar", "onnx"), default=None, help="覆盖配置中的 inference.method")
    parser.add_argument("--json-out", type=Path, default=None, help="将结果写入 JSON 文件")
    args = parser.parse_args()

    cfg = load_config(args.config)
    method = args.method or cfg.get("inference", {}).get("method", "haar")
    src = args.source.resolve()

    try:
        if src.is_dir():
            results = [infer_image(p, method, cfg) for p in _list_images(src)]
            payload: dict[str, Any] = {"results": results}
        else:
            payload = infer_image(src, method, cfg)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

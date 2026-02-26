from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import torch
from efficientnet_pytorch import EfficientNet
from torch import nn

from . import deepfake_image as di
from .deepfake_image import _create_xception_model, _load_forgiving

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

_EFFICIENTNET: nn.Module | None = None
_XCEPTION: nn.Module | None = None


def _load_model() -> None:
    global _EFFICIENTNET, _XCEPTION
    if _EFFICIENTNET is not None:
        return

    # Match the EfficientNet variant and number of classes used for the
    # image checkpoint (B4, 2 classes: real / fake).
    model_name = "efficientnet-b4"
    _EFFICIENTNET = EfficientNet.from_name(model_name, num_classes=2)

    efficientnet_ckpt = MODELS_DIR / "efficientnet_model.pth"
    if not efficientnet_ckpt.exists():
        efficientnet_ckpt = MODELS_DIR / "deepfake_video_efficientnet.pth"

    if efficientnet_ckpt.exists():
        _load_forgiving(
            _EFFICIENTNET,
            efficientnet_ckpt,
            prefixes=("module.", "backbone.efficientnet.", "backbone."),
        )

    _XCEPTION = None
    xception_ckpt = MODELS_DIR / "xception_model.pth"
    if xception_ckpt.exists():
        _XCEPTION = _create_xception_model()
        if _XCEPTION is not None:
            _load_forgiving(
                _XCEPTION,
                xception_ckpt,
                prefixes=("module.", "backbone.xception.", "backbone."),
            )

    _EFFICIENTNET.eval()
    if _XCEPTION is not None:
        _XCEPTION.eval()


def _sample_frames(video_path: Path, num_frames: int = 24):
    cap = cv2.VideoCapture(str(video_path))
    frames: list[dict[str, Any]] = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    indices = sorted({min(frame_count - 1, int(i * frame_count / num_frames)) for i in range(num_frames)})

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # OpenCV gives BGR; convert to RGB for PIL / processors
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        time_s = float(idx / fps)
        frames.append({"frame_index": int(idx), "time_s": time_s, "rgb": rgb})

    cap.release()
    return frames


def analyze_video(video_path: Path) -> dict:
    """
    Analyze a video for deepfake likelihood by sampling frames.
    """
    try:
        _load_model()
        frames = _sample_frames(video_path)
        if not frames:
            raise RuntimeError("No frames could be sampled from video.")

        di._load_model()
        hf_proc = getattr(di, "_HF_PROCESSOR", None)
        hf_model = getattr(di, "_HF_MODEL", None)

        frame_probs: list[dict[str, Any]] = []
        heatmap_file: str | None = None

        # Use the pretrained HF deepfake detector frame-by-frame.
        if hf_proc is not None and hf_model is not None:
            from PIL import Image

            idx_fake = di._deepfake_class_index_from_config(hf_model.config.id2label)

            best = None  # (prob, frame_dict, attentions)
            for f in frames:
                img = Image.fromarray(f["rgb"]).convert("RGB")
                inputs = hf_proc(images=img, return_tensors="pt")
                with torch.no_grad():
                    outputs = hf_model(**inputs, output_attentions=True)
                    probs = torch.softmax(outputs.logits, dim=-1)[0]
                p_fake = float(probs[idx_fake].item())
                frame_probs.append(
                    {"frame": f["frame_index"], "time_s": round(float(f["time_s"]), 2), "prob": round(p_fake, 4)}
                )
                if best is None or p_fake > best[0]:
                    best = (p_fake, img, getattr(outputs, "attentions", None))

            # Overall: use max probability across sampled frames (worst-case).
            deepfake_prob = float(max(fp["prob"] for fp in frame_probs))
            confidence = float(max(fp["prob"] for fp in frame_probs) * 100.0)

            if best is not None:
                heatmap_file = di._save_attention_heatmap_vit(best[1], best[2], "vid_heatmap")

            risk = max(0.0, min(100.0, deepfake_prob * 100.0))
            details = (
                "Video analysed frame-by-frame using pretrained deepfake classifier (ViT).\n"
                f"Max deepfake probability across sampled frames: {deepfake_prob:.2f}\n"
                f"Frames analysed: {len(frame_probs)}\n"
                f"File: {video_path.name}"
            )

            result = {
                "risk": float(risk),
                "confidence": float(confidence),
                "details": details,
                "frame_probs": frame_probs,
            }
            if heatmap_file:
                result["heatmap_file"] = heatmap_file
            return result
        raise RuntimeError(
            "No deepfake detector model available (missing local weights and HF fallback failed)."
        )
    except Exception as exc:
        risk = 50.0
        confidence = 30.0
        details = (
            f"Deepfake video model not fully configured or failed with error: {exc}\n"
            "Default heuristic result returned."
        )

    return {
        "risk": float(risk),
        "confidence": float(confidence),
        "details": details,
    }
from __future__ import annotations

import math
from pathlib import Path
from uuid import uuid4

import cv2
import torch
from torch import nn

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"

_XCEPTION: nn.Module | None = None
_CONVNEXT: nn.Module | None = None
_HF_PROCESSOR = None
_HF_MODEL = None

# A stronger fallback model (downloads from Hugging Face the first time).
_HF_MODEL_NAME = "dima806/deepfake_vs_real_image_detection"


def _unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _strip_prefixes(state_dict: dict, prefixes: tuple[str, ...]) -> dict:
    out: dict = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


def _load_forgiving(model: nn.Module, checkpoint_path: Path, prefixes: tuple[str, ...]) -> None:
    """
    Load a checkpoint while:
    - Unwrapping common container keys
    - Stripping known prefixes
    - Dropping any parameters whose shapes don't match the target model
    This avoids EfficientNet size-mismatch crashes if the architecture or
    implementation details differ slightly from the original training setup.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _unwrap_state_dict(ckpt)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(state_dict)}")
    state_dict = _strip_prefixes(state_dict, prefixes)

    target_state = model.state_dict()
    filtered_state = {}
    for k, v in state_dict.items():
        if k in target_state and target_state[k].shape == v.shape:
            filtered_state[k] = v

    missing = set(target_state.keys()) - set(filtered_state.keys())
    if missing:
        # Optional: could log this instead of printing.
        print(f"[deepfake_image] Skipped {len(missing)} mismatched / missing keys when loading {checkpoint_path.name}")

    model.load_state_dict(filtered_state, strict=False)


def _create_xception_model() -> nn.Module | None:
    """
    Tries to create a timm Xception model if available.
    If your project uses a custom Xception implementation, replace this function accordingly.
    """
    try:
        import timm  # type: ignore
    except Exception:
        return None

    for name in ("xception", "xception41", "xception65", "xception71"):
        try:
            m = timm.create_model(name, pretrained=False, num_classes=2)
            m.eval()
            return m
        except Exception:
            continue
    return None


def _create_convnext_model() -> nn.Module | None:
    try:
        import timm  # type: ignore
    except Exception:
        return None

    # Keep it reasonably fast on CPU.
    for name in ("convnext_tiny", "convnext_small", "convnext_base"):
        try:
            m = timm.create_model(name, pretrained=False, num_classes=2)
            m.eval()
            return m
        except Exception:
            continue
    return None


def _load_model() -> None:
    global _XCEPTION, _CONVNEXT, _HF_PROCESSOR, _HF_MODEL
    if _XCEPTION is not None or _CONVNEXT is not None or _HF_MODEL is not None:
        return

    # 1) Try to load local Xception / ConvNeXt checkpoints if they exist.
    #    We first look in models/, then fall back to the project root (BASE_DIR)
    #    because some setups keep the .pth files alongside app.py.
    xception_ckpt = MODELS_DIR / "xception_model.pth"
    if not xception_ckpt.exists():
        xception_ckpt = BASE_DIR / "xception_model.pth"

    convnext_ckpt = MODELS_DIR / "convnext_model.pth"
    if not convnext_ckpt.exists():
        convnext_ckpt = BASE_DIR / "convnext_model.pth"

    if xception_ckpt.exists():
        _XCEPTION = _create_xception_model()
        if _XCEPTION is not None:
            _load_forgiving(
                _XCEPTION,
                xception_ckpt,
                prefixes=("module.", "backbone.xception.", "backbone."),
            )
            _XCEPTION.eval()

    if convnext_ckpt.exists():
        _CONVNEXT = _create_convnext_model()
        if _CONVNEXT is not None:
            _load_forgiving(
                _CONVNEXT,
                convnext_ckpt,
                prefixes=("module.", "backbone.convnext.", "backbone."),
            )
            _CONVNEXT.eval()

    # 2) Always try to load the HF ViT detector as a third model.
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        _HF_PROCESSOR = AutoImageProcessor.from_pretrained(_HF_MODEL_NAME)
        _HF_MODEL = AutoModelForImageClassification.from_pretrained(_HF_MODEL_NAME)
        _HF_MODEL.eval()
    except Exception:
        _HF_PROCESSOR = None
        _HF_MODEL = None


def _deepfake_class_index_from_config(id2label: dict) -> int:
    """
    Infer which class index corresponds to 'fake' / 'deepfake'.
    Falls back to 1 if unclear.
    """
    best_idx = None
    for k, v in id2label.items():
        label = str(v).lower()
        if "deepfake" in label or label in {"fake", "spoof", "manipulated"}:
            best_idx = int(k)
            break
    return best_idx if best_idx is not None else 1


def _save_attention_heatmap_vit(pil_rgb_image, attentions, out_prefix: str) -> str | None:
    """
    Create a simple ViT attention heatmap (CLS -> patches) from the last layer
    and save an overlay PNG in uploads/.
    Returns the filename (not full path) or None on failure.
    """
    import numpy as np

    if not attentions:
        return None

    last = attentions[-1]  # (B, heads, tokens, tokens)
    att = last.mean(dim=1)[0]  # (tokens, tokens)
    cls_to_patches = att[0, 1:]  # (num_patches,)
    grid = int(math.sqrt(cls_to_patches.shape[0]))
    if grid * grid != cls_to_patches.shape[0]:
        return None

    heat = cls_to_patches.reshape(grid, grid).detach().cpu().numpy()
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()

    # PIL RGB -> numpy BGR (for OpenCV)
    img_bgr = cv2.cvtColor(np.array(pil_rgb_image), cv2.COLOR_RGB2BGR)

    heat_resized = cv2.resize(heat, (img_bgr.shape[1], img_bgr.shape[0]))
    heat_uint8 = (heat_resized * 255).astype("uint8")
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, colored, 0.4, 0)

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{out_prefix}_{uuid4().hex}.png"
    out_path = UPLOADS_DIR / filename
    cv2.imwrite(str(out_path), overlay)
    return filename


def _pil_from_path(image_path: Path):
    from PIL import Image

    return Image.open(str(image_path)).convert("RGB")


def _preprocess_rgb_np(rgb, size: int) -> torch.Tensor:
    """
    rgb: HxWx3 uint8 (RGB)
    returns: 1x3xSxS float tensor (ImageNet normalized)
    """
    img = cv2.resize(rgb, (size, size))
    img = img.astype("float32") / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = torch.from_numpy(img).permute(2, 0, 1)
    t = (t - mean) / std
    return t.unsqueeze(0)


def _preprocess_image(image_path: Path):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = img.transpose(2, 0, 1)
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor


def analyze_image(image_path: Path) -> dict:
    """
    Analyze an image for deepfake likelihood.
    """
    try:
        _load_model()

        # Coordinate multiple models and average their fake probabilities.
        probs_by_model: dict[str, float] = {}
        heatmap_filename = None

        # 1) HF ViT deepfake detector (if available)
        if _HF_PROCESSOR is not None and _HF_MODEL is not None:
            img = _pil_from_path(image_path)
            inputs = _HF_PROCESSOR(images=img, return_tensors="pt")

            with torch.no_grad():
                outputs = _HF_MODEL(**inputs, output_attentions=True)
                probs = torch.softmax(outputs.logits, dim=-1)[0]

            idx_fake = _deepfake_class_index_from_config(_HF_MODEL.config.id2label)
            vit_prob = float(probs[idx_fake].item())
            probs_by_model["ViT"] = vit_prob

            heatmap_filename = _save_attention_heatmap_vit(
                img, getattr(outputs, "attentions", None), "img_heatmap"
            )

        # 2) Local Xception (if weights were provided)
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is not None:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if _XCEPTION is not None:
                with torch.no_grad():
                    t = _preprocess_rgb_np(rgb, 299)
                    p = torch.softmax(_XCEPTION(t), dim=-1)[0]
                    x_prob = float(p[1].item() if p.numel() > 1 else p[0].item())
                probs_by_model["Xception"] = x_prob

            if _CONVNEXT is not None:
                with torch.no_grad():
                    t = _preprocess_rgb_np(rgb, 224)
                    p = torch.softmax(_CONVNEXT(t), dim=-1)[0]
                    c_prob = float(p[1].item() if p.numel() > 1 else p[0].item())
                probs_by_model["ConvNeXt"] = c_prob

        if not probs_by_model:
            raise RuntimeError(
                "No deepfake detector model available (missing local weights and HF fallback failed)."
            )

        # Weighted average fake probability: trust ViT more than generic CNNs.
        raw_weights: dict[str, float] = {}
        for name in probs_by_model.keys():
            if name == "ViT":
                raw_weights[name] = 0.6
            elif name == "Xception":
                raw_weights[name] = 0.2
            elif name == "ConvNeXt":
                raw_weights[name] = 0.2
            else:
                raw_weights[name] = 0.2
        total_w = sum(raw_weights.values())
        weights = {k: v / total_w for k, v in raw_weights.items()}

        avg_prob = float(
            sum(probs_by_model[m] * weights[m] for m in probs_by_model.keys())
        )
        risk = max(0.0, min(100.0, avg_prob * 100.0))
        confidence = max(probs_by_model.values()) * 100.0

        lines = [
            "Image analysed using deepfake ensemble (average across models).",
        ]
        for name, p in probs_by_model.items():
            w = weights.get(name, 0.0)
            lines.append(f"{name} fake probability: {p:.2f} (weight {w:.2f})")
        lines.append(f"Weighted average fake probability: {avg_prob:.2f}")
        lines.append(f"File: {image_path.name}")

        details = "\n".join(lines)

        result = {
            "risk": float(risk),
            "confidence": float(confidence),
            "details": details,
        }
        if heatmap_filename:
            result["heatmap_file"] = heatmap_filename
        return result
    except Exception as exc:
        risk = 50.0
        confidence = 30.0
        details = (
            f"Deepfake image model not fully configured or failed with error: {exc}\n"
            "Default heuristic result returned."
        )

    return {
        "risk": float(risk),
        "confidence": float(confidence),
        "details": details,
    }
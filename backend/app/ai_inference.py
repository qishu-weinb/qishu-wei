"""Local inference for the trained BUS-BRA + BUSI model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from PIL import Image

from .ai_model import MultiTaskEfficientNet
from .config import MODEL_INPUT_SIZE, MODEL_PATH, MODEL_VERSION


LABELS = ("normal", "benign", "malignant")
_inference_lock = Lock()


@lru_cache(maxsize=1)
def _load_model() -> tuple[MultiTaskEfficientNet, torch.device]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型权重不存在: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskEfficientNet(pretrained=False).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, device


def model_status() -> dict:
    status = {
        "configured": MODEL_PATH.exists(),
        "modelVersion": MODEL_VERSION,
        "modelPath": str(MODEL_PATH),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    try:
        _load_model()
        status["ready"] = True
    except Exception as exc:  # health endpoint should expose the real cause
        status["ready"] = False
        status["error"] = str(exc)
    return status


def infer_image(image_path: str | Path) -> dict:
    model, device = _load_model()
    image = Image.open(image_path).convert("L").resize(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BILINEAR
    )
    image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
    image_tensor = image_tensor.div(255.0).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    with _inference_lock, torch.inference_mode():
        logits, mask_logits = model(image_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        mask = (torch.sigmoid(mask_logits)[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
    index = int(probabilities.argmax())
    return {
        "result": LABELS[index],
        "confidence": float(probabilities[index]),
        "probabilities": {label: float(value) for label, value in zip(LABELS, probabilities)},
        "mask": mask,
        "modelVersion": MODEL_VERSION,
        "device": str(device),
    }

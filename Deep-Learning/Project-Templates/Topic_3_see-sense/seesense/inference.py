"""Preprocessing, prediction, and Grad-CAM overlays for uploaded images.

Only numpy + pillow + torch. The overlay is rendered here (server side) so the
Streamlit UI stays a thin client that just base64-decodes a PNG.
"""

import base64
import hashlib
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from . import CLASSES, MEAN, STD

_MEAN = torch.tensor(MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(STD).view(1, 3, 1, 1)

MAX_OVERLAY_SIDE = 1024   # keep the base64 PNG in the JSON response small


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def open_image(data: bytes) -> Image.Image:
    """Decode upload bytes into an upright RGB image (raises ValueError if it is not an image)."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # PIL raises several unrelated types
        raise ValueError(f"not a readable image: {exc}") from exc
    return ImageOps.exif_transpose(img).convert("RGB")


def center_square(img: Image.Image) -> tuple[int, int, int, int]:
    """The (left, top, right, bottom) box ImageOps.fit keeps when it crops to a square."""
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return left, top, left + side, top + side


def preprocess(img: Image.Image, size: int = 32) -> torch.Tensor:
    """Any photo -> uint8 tensor (3, 32, 32): center-crop to a square, antialiased downsample."""
    small = img.crop(center_square(img)).resize((size, size), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(small).copy()).permute(2, 0, 1)


def to_model_input(x_uint8: torch.Tensor) -> torch.Tensor:
    return (x_uint8.float() / 255.0 - _MEAN) / _STD


@torch.no_grad()
def predict_proba(model, x_uint8: torch.Tensor, tta: bool = True) -> np.ndarray:
    """Softmax probabilities for a uint8 batch (N, 3, 32, 32); `tta` averages with the mirror image."""
    model.eval()
    xb = to_model_input(x_uint8)
    logits = model(xb)
    if tta:
        logits = (logits + model(xb.flip(3))) / 2
    return F.softmax(logits, dim=1).numpy()


def grad_cam(model, x_uint8: torch.Tensor, class_idx: int) -> np.ndarray:
    """Grad-CAM heatmap (8x8, values in [0, 1]) for one uint8 image (3, 32, 32).

    alpha_k = mean over (i, j) of d y_c / d A_k[i, j];  L_c = ReLU(sum_k alpha_k A_k)
    """
    model.eval()
    x = to_model_input(x_uint8.unsqueeze(0))
    with torch.enable_grad():
        feats = model.features(x)
        feats.retain_grad()
        logits = model.head(model.stage3[-2:](feats))
        model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(1))[0].detach()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam.numpy()


def jet(v: np.ndarray) -> np.ndarray:
    """The 'jet' colormap for values in [0, 1] -> RGB floats in [0, 1] (no matplotlib needed)."""
    v = v[..., None]
    rgb = np.concatenate([1.5 - np.abs(4 * v - 3), 1.5 - np.abs(4 * v - 2), 1.5 - np.abs(4 * v - 1)], -1)
    return np.clip(rgb, 0, 1)


def blend(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heat = Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.Resampling.BILINEAR)
    colored = jet(np.asarray(heat) / 255.0)
    base = np.asarray(img.convert("RGB")) / 255.0
    return Image.fromarray(np.uint8(((1 - alpha) * base + alpha * colored) * 255))


def overlay_full(img: Image.Image, cam: np.ndarray) -> Image.Image:
    """Grad-CAM over the original photo, same width x height (scaled down past MAX_OVERLAY_SIDE).

    The heatmap covers the center square the model actually saw; the cropped-away
    margins are darkened, which shows visitors what the CNN never looked at.
    """
    scale = min(1.0, MAX_OVERLAY_SIDE / max(img.size))
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    box = center_square(img)
    out = Image.fromarray(np.uint8(np.asarray(img) * 0.35))
    out.paste(blend(img.crop(box), cam), box[:2])
    return out


def model_view(x_uint8: torch.Tensor, cam: np.ndarray, size: int = 224) -> Image.Image:
    """What the model sees: the 32x32 input, enlarged with crisp pixels, with Grad-CAM on top."""
    small = Image.fromarray(x_uint8.permute(1, 2, 0).numpy())
    return blend(small.resize((size, size), Image.Resampling.NEAREST), cam)


def png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def explain(model, img: Image.Image, tta: bool = True, top_k: int = 5) -> dict:
    """Classify one image and build both Grad-CAM views."""
    x = preprocess(img)
    probs = predict_proba(model, x.unsqueeze(0), tta=tta)[0]
    order = np.argsort(probs)[::-1]
    pred = int(order[0])
    cam = grad_cam(model, x, pred)
    return dict(
        label=pred,
        class_name=CLASSES[pred],
        confidence=float(probs[pred]),
        probabilities={c: float(p) for c, p in zip(CLASSES, probs)},
        top5=[{"class_name": CLASSES[i], "probability": float(probs[i])} for i in order[:top_k]],
        cam=cam,
        overlay=overlay_full(img, cam),
        model_view=model_view(x, cam),
    )

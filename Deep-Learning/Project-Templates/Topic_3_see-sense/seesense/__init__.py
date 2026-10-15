"""See-Sense model code: the CIFAR-10 CNNs, preprocessing, and Grad-CAM.

Inference-only copy of the training project (`AIT-204-CNN/cifar10_project/cifar`),
trimmed so the Render image needs only torch + numpy + pillow (no torchvision,
no matplotlib). Training happens offline; see scripts/export_models.py.
"""

from pathlib import Path

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# Per-channel statistics of the CIFAR-10 training set (RGB, in [0, 1] units).
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

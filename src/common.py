"""Utilidades compartidas por los scripts de entrenamiento e inferencia."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "YOLOv8s-50e.pt"

# Colores en formato BGR de OpenCV: jugadores en azul, balón en verde.
CLASS_COLORS = {"player": (255, 0, 0), "ball": (0, 255, 0)}
FALLBACK_COLOR = (255, 255, 0)


def resolve_path(value: str | Path) -> Path:
    """Devuelve una ruta absoluta; las relativas se interpretan desde la raíz del repo."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def resolve_device(value: str = "auto") -> str:
    """Con `auto`, selecciona CUDA -> MPS -> CPU según lo disponible."""
    if value != "auto":
        return value
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def color_for(class_name: str) -> tuple[int, int, int]:
    """Color BGR asociado a una clase del modelo."""
    return CLASS_COLORS.get(class_name.strip().lower(), FALLBACK_COLOR)


def draw_detections(frame, result, conf: float):
    """Dibuja sobre `frame` las detecciones de `result` que superan `conf`."""
    import cv2

    for box in result.boxes:
        score = float(box.conf[0])
        if score < conf:
            continue
        class_name = result.names[int(box.cls[0])]
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        color = color_for(class_name)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{class_name} {score:.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame

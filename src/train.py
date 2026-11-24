"""Entrena un detector YOLO con el dataset de voleibol.

Ejemplo:
    python src/train.py --data data/data.yaml --epochs 50
    python src/train.py --data data/data.yaml --model yolo26s.pt --epochs 50
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import ROOT, resolve_device, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help="Modelo base: nombre de Ultralytics (yolov8s.pt, yolo26s.pt) o ruta a un .pt local.",
    )
    parser.add_argument("--data", default="data/data.yaml", help="Ruta al YAML del dataset.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        default="auto",
        help="Dispositivo: auto, cpu, mps, cuda o índice de GPU (por defecto, auto).",
    )
    parser.add_argument("--name", default="volleyball", help="Nombre del run en runs/detect/.")
    return parser.parse_args()


def model_reference(value: str) -> str:
    """Deja los nombres oficiales intactos; resuelve las rutas locales a .pt."""
    path = Path(value).expanduser()
    if path.suffix.lower() == ".pt" and path.parent != Path("."):
        return str(resolve_path(path))
    return value


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    data = resolve_path(args.data)
    if not data.is_file():
        raise SystemExit(
            f"No se encontró el dataset: {data}\n"
            "Copia data/data.yaml.example a data/data.yaml y ajusta las rutas."
        )

    device = resolve_device(args.device)
    print(f"Entrenando con modelo={args.model!r}, datos={data}, device={device}")
    model = YOLO(model_reference(args.model))
    model.info()

    started = time.perf_counter()
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=device,
        name=args.name,
    )
    print(f"Entrenamiento completado en {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()

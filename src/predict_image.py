"""Detecta jugadores y balón en una imagen.

Ejemplo:
    python src/predict_image.py --source assets/labeled.jpg --output assets/predicted_new.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_MODEL, draw_detections, resolve_device, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Ruta a la imagen de entrada.")
    parser.add_argument("--output", help="Ruta de la imagen de salida (opcional).")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Pesos del modelo.")
    parser.add_argument("--conf", type=float, default=0.5, help="Umbral de confianza.")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda o índice de GPU.")
    parser.add_argument("--show", action="store_true", help="Muestra la imagen en una ventana.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import cv2
    from ultralytics import YOLO

    source = resolve_path(args.source)
    if not source.is_file():
        raise SystemExit(f"No se encontró la imagen: {source}")

    frame = cv2.imread(str(source))
    if frame is None:
        raise SystemExit(f"No se pudo leer la imagen: {source}")

    model = YOLO(str(resolve_path(args.model)))
    result = model(frame, conf=args.conf, device=resolve_device(args.device), verbose=False)[0]
    frame = draw_detections(frame, result, args.conf)

    if args.output:
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), frame):
            raise SystemExit(f"No se pudo escribir la imagen: {output}")
        print(f"Imagen guardada en {output}")

    if args.show:
        cv2.imshow("Deteccion", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not args.output and not args.show:
        print("Usa --output para guardar el resultado o --show para visualizarlo.")


if __name__ == "__main__":
    main()

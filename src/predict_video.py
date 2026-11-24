"""Detecta jugadores y balón en un vídeo o en la webcam.

Ejemplo:
    python src/predict_video.py --source "ruta/al/partido.mp4"
    python src/predict_video.py --source 0 --show
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_MODEL, draw_detections, resolve_device, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="Ruta al vídeo de entrada o '0' para usar la webcam.",
    )
    parser.add_argument(
        "--output",
        help="Ruta del vídeo de salida (por defecto, <entrada>_detected.mp4).",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Pesos del modelo.")
    parser.add_argument("--conf", type=float, default=0.5, help="Umbral de confianza.")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda o índice de GPU.")
    parser.add_argument("--codec", default="mp4v", help="Codec de OpenCV (mp4v, avc1, ...).")
    parser.add_argument("--show", action="store_true", help="Muestra el vídeo mientras se procesa.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import cv2
    from ultralytics import YOLO

    webcam = args.source.isdigit()
    source = int(args.source) if webcam else str(resolve_path(args.source))
    if not webcam and not Path(source).is_file():
        raise SystemExit(f"No se encontró el vídeo: {source}")

    if args.output:
        output = resolve_path(args.output)
    elif webcam:
        output = resolve_path("webcam_detected.mp4")
    else:
        original = Path(source)
        output = original.with_name(f"{original.stem}_detected{original.suffix or '.mp4'}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la entrada de vídeo: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*args.codec), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"No se pudo crear el vídeo de salida: {output} (prueba otro --codec)")

    model = YOLO(str(resolve_path(args.model)))
    device = resolve_device(args.device)
    frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = model(frame, conf=args.conf, device=device, verbose=False)[0]
            writer.write(draw_detections(frame, result, args.conf))
            frames += 1
            if args.show:
                cv2.imshow("Deteccion", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Procesados {frames} fotogramas -> {output}")


if __name__ == "__main__":
    main()

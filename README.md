# Reconocimiento de Objetos en Voleibol usando YOLOv8

Este proyecto presenta el entrenamiento y evaluación de modelos YOLOv8 para detectar jugadores y balones en partidos de voleibol. Se comparan las variantes **YOLOv8n (Nano)** y **YOLOv8s (Small)** para determinar el equilibrio óptimo entre velocidad y precisión.

## 📋 Introducción
El objetivo es implementar un sistema de detección en tiempo real útil para el análisis deportivo y el seguimiento de partidos. El proyecto abarca desde la creación de un dataset personalizado hasta la prueba final en video.

## 📊 Dataset
Debido a la falta de datasets específicos, se creó uno propio utilizando:
* **Fuente:** Videos de YouTube.
* **Herramienta de etiquetado:** Roboflow.
* **Contenido:** 439 imágenes tras post-procesamiento. (https://universe.roboflow.com/antonyma/volleyball-video/dataset/2)
* **Clases:** `Player` (Jugador) y `Ball` (Balón).

## ⚙️ Configuración del Entorno
* **Hardware:** Apple M2 (usando aceleración `mps`).
* **Software:** Python 3.9.20, Ultralytics 8.3.40, Torch 2.5.1.
* **Entorno:** Anaconda3.

## 🚀 Entrenamiento y Resultados

Se realizaron dos entrenamientos principales con imágenes de 640x640:

| Modelo | Épocas | mAP@0.5 (Global) | FPS | Conclusión |
| :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n** | 100 | 11.1% | 0.54 | Eficiente pero con baja precisión en el balón. |
| **YOLOv8s** | 50 | 91.0% | 1.36 | Excelente balance y robustez en la detección. |

### Comparativa de Precisión
* El modelo **YOLOv8s** demostró ser significativamente superior, logrando un **98.9% de mAP** para la clase `Player` y **88.3%** para la clase `Ball`.
* Se observó que el modelo ligero (Nano) tiene dificultades para generalizar y detectar objetos pequeños como el balón.

## 💻 Uso del Código
El repositorio incluye dos scripts principales:
* `yoloV8-Image.py`: Script para el entrenamiento inicial del modelo usando `data.yaml`.
* `yoloV8-video.py`: Script para realizar inferencia en archivos de video, dibujando cajas delimitadoras para jugadores (azul) y balones (verde).

## 💡 Conclusiones
El experimento demuestra que, aunque los modelos ligeros son rápidos, requieren datasets más diversos y grandes para tareas de alta precisión. YOLOv8s es la opción recomendada para este caso de uso deportivo.

---
**Autor:** Antonio Jesús Angulo Magdaleno
**Fecha:** Diciembre 2024 

# Reconocimiento de Objetos en Voleibol usando YOLOv8

[cite_start]Este proyecto presenta el entrenamiento y evaluación de modelos YOLOv8 para detectar jugadores y balones en partidos de voleibol[cite: 93, 99]. [cite_start]Se comparan las variantes **YOLOv8n (Nano)** y **YOLOv8s (Small)** para determinar el equilibrio óptimo entre velocidad y precisión[cite: 112, 115].

## 📋 Introducción
[cite_start]El objetivo es implementar un sistema de detección en tiempo real útil para el análisis deportivo y el seguimiento de partidos[cite: 100]. [cite_start]El proyecto abarca desde la creación de un dataset personalizado hasta la prueba final en video[cite: 99].

## 📊 Dataset
Debido a la falta de datasets específicos, se creó uno propio utilizando:
* [cite_start]**Fuente:** Videos de YouTube[cite: 105].
* [cite_start]**Herramienta de etiquetado:** Roboflow[cite: 105].
* [cite_start]**Contenido:** 439 imágenes tras post-procesamiento[cite: 107, 108].
* [cite_start]**Clases:** `Player` (Jugador) y `Ball` (Balón)[cite: 103].

## ⚙️ Configuración del Entorno
* [cite_start]**Hardware:** Apple M2 (usando aceleración `mps`)[cite: 110, 114].
* [cite_start]**Software:** Python 3.9.20, Ultralytics 8.3.40, Torch 2.5.1[cite: 111].
* [cite_start]**Entorno:** Anaconda3[cite: 111].

## 🚀 Entrenamiento y Resultados

[cite_start]Se realizaron dos entrenamientos principales con imágenes de 640x640[cite: 114, 116]:

| Modelo | Épocas | mAP@0.5 (Global) | FPS | Conclusión |
| :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n** | 100 | 11.1% | 0.54 | [cite_start]Eficiente pero con baja precisión en el balón[cite: 122, 179]. |
| **YOLOv8s** | 50 | 91.0% | 1.36 | [cite_start]Excelente balance y robustez en la detección[cite: 295, 306]. |

### Comparativa de Precisión
* [cite_start]El modelo **YOLOv8s** demostró ser significativamente superior, logrando un **98.9% de mAP** para la clase `Player` y **88.3%** para la clase `Ball`[cite: 295].
* [cite_start]Se observó que el modelo ligero (Nano) tiene dificultades para generalizar y detectar objetos pequeños como el balón[cite: 124, 85].

## 💻 Uso del Código
El repositorio incluye dos scripts principales:
* `yoloV8-Image.py`: Script para el entrenamiento inicial del modelo usando `data.yaml`.
* `yoloV8-video.py`: Script para realizar inferencia en archivos de video, dibujando cajas delimitadoras para jugadores (azul) y balones (verde).

## 💡 Conclusiones
[cite_start]El experimento demuestra que, aunque los modelos ligeros son rápidos, requieren datasets más diversos y grandes para tareas de alta precisión[cite: 87, 88]. [cite_start]YOLOv8s es la opción recomendada para este caso de uso deportivo[cite: 86].

---
[cite_start]**Autor:** Antonio Jesús Angulo Magdaleno [cite: 97]  
[cite_start]**Fecha:** Diciembre 2024 [cite: 96]

# Defect Detection using SAM2  
**TFM - Universidad Camilo José Cela (2025)**  
*Natalia Peinado Verhoeven · natalia.peinado@alumno.ucjc.edu*

---
# SAM2 - Segment Anything for Industrial Defect Detection (Severstal Dataset)

Este proyecto adapta y entrena el modelo **SAM2 (Segment Anything Model v2)** para tareas de segmentación semántica en imágenes industriales con defectos, utilizando el dataset **Severstal**. Incluye scripts para entrenamiento, evaluación, análisis de clases y exploración de subconjuntos.

---

## Estructura de Carpetas
sam2/
├── configs/ # Configuración de modelos SAM2
├── dataloaders/ # Scripts de carga y preprocesado de datos
├── datasets/severstal/ # Dataset Severstal organizado en splits
│ ├── train_split/ # 70% datos de entrenamiento
│ ├── test_split/ # 30% datos de test
│ ├── train_25/ ... # Subconjuntos para curva de aprendizaje
├── models/ # Checkpoints del modelo base y fine-tuned
├── logs/ # Métricas registradas en CSV
├── src/
│ ├── train.py # Script de entrenamiento
│ ├── evaluate.py # Evaluación completa y por imagen
│ ├── main.py # Punto de entrada principal
│ ├── utils.py # Funciones auxiliares
│ ├── split_dataset.py # Generación de splits
│ ├── metrics_logger.py # Registro de métricas
│ ├── explore_classes.py # Análisis de clases y visualización


---

##  Entrenamiento

Entrena SAM2 con el conjunto completo (`train_split`):

```bash
CUDA_VISIBLE_DEVICES=0 python src/main.py train \
  --dataset severstal \
  --checkpoint models/sam2_hiera_large.pt \
  --config configs/sam2/sam2_hiera_l.yaml \
  --model_name fine_tuned_sam2_severstal_split \
  --steps 3000 \
  --annotation_format bitmap

Solo se guarda el modelo con mejor IoU

##  Evaluación
Evalúa el modelo entrenado sobre test_split:

CUDA_VISIBLE_DEVICES=0 python src/main.py eval_full_test \
  --dataset severstal \
  --checkpoint models/sam2_hiera_large.pt \
  --config configs/sam2/sam2_hiera_l.yaml \
  --fine_tuned_checkpoint models/severstal/fine_tuned_sam2_severstal_split_best_stepXXXX_iou0.XX.torch \
  --annotation_format bitmap

Los resultados se guardan en: logs/eval_metrics.csv

## Registro de métricas
Durante el entrenamiento y la evaluación se guarda automáticamente:

- IoU medio
- IoU@50, @75, @95 (en evaluación)
- Nombre del modelo, subset y step

Todo se almacena en CSVs dentro de logs/.

## Análisis de subconjuntos
Para entrenar sobre datasets reducidos (train_25, train_50, etc.):

CUDA_VISIBLE_DEVICES=0 python src/main.py train \
  --dataset severstal25 \
  --checkpoint models/sam2_hiera_large.pt \
  --config configs/sam2/sam2_hiera_l.yaml \
  --model_name fine_tuned_sam2_25 \
  --steps 1000 \
  --annotation_format bitmap

## Análisis de clases
Explora cuántas clases existen y cuántas imágenes tiene cada una:

python src/explore_classes.py

Esto genera:

- Conteo por clase
- Número de imágenes con más de una clase
- Ejemplos visuales por clase

## Requisitos

- Python ≥ 3.9
- PyTorch ≥ 1.12
- OpenCV, NumPy, Matplotlib
- tqdm, scikit-learn

pip install -r requirements.txt

## Créditos
Este proyecto se basa en el modelo SAM2 (Segment Anything v2) y adapta su uso a defectos industriales con anotaciones tipo bitmap comprimidas.

## Licencia
Este repositorio se proporciona solo con fines académicos y de investigación. Consulta la licencia original de SAM2 para su uso comercial.

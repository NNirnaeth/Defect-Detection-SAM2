import os
import json
import base64
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from utils import decode_bitmap_to_mask
from evaluate import decode_bitmap_from_entry

# Rutas
ANNOTATIONS_PATH = "/home/ptp/sam2/datasets/severstal/total_dataset/ann"
IMAGES_PATH = "/home/ptp/sam2/datasets/severstal/total_dataset/img"

# Inicializar contadores
class_counts = defaultdict(int)
image_class_sets = defaultdict(set)
example_images = {}

# Recorrer anotaciones
for filename in tqdm(os.listdir(ANNOTATIONS_PATH)):
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(ANNOTATIONS_PATH, filename)
    with open(json_path, "r") as f:
        data = json.load(f)

    img_name = filename.replace(".json", ".jpg")
    img_path = os.path.join(IMAGES_PATH, img_name)

    if not os.path.exists(img_path):
        continue

    img = np.array(Image.open(img_path).convert("RGB"))
    class_titles = set()


    has_bitmap = any(obj.get("geometryType") == "bitmap" for obj in data.get("objects", []))
    if not has_bitmap:
        continue

    class_titles = set()
    for obj in data["objects"]:
        label = obj.get("classTitle")
        class_titles.add(label)
        class_counts[label] += 1

    # Guardar ejemplo visual (una vez por clase)
    if class_titles and all(label not in example_images for label in class_titles):
        mask = decode_bitmap_from_entry(data)
        if mask is None or mask.sum() == 0:
            continue
        overlay = img.copy()
        overlay[mask > 0] = [255, 0, 0]  # rojo
        for label in class_titles:
            if label not in example_images:
                example_images[label] = overlay.copy()

    # Guardar cuántas clases tiene esta imagen
    image_class_sets[img_name] = class_titles

# Mostrar conteo por clase
print("\n Conteo de defectos por clase:")
for label, count in sorted(class_counts.items()):
    print(f" - {label}: {count} instancias")

# Cuántas imágenes tienen múltiples clases
multi_class_count = sum(1 for v in image_class_sets.values() if len(v) > 1)
print(f"\n Imágenes con más de una clase de defecto: {multi_class_count} de {len(image_class_sets)}")

# Visualización por clase
if len(example_images) == 0:
    print(" No se encontraron ejemplos visuales con máscaras válidas.")
else:
    print("\n Ejemplos visuales (1 por clase):")
    fig, axs = plt.subplots(1, len(example_images), figsize=(5 * len(example_images), 5))
    if len(example_images) == 1:
        axs = [axs]
    for ax, (label, vis_img) in zip(axs, example_images.items()):
        ax.imshow(vis_img)
        ax.set_title(label)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

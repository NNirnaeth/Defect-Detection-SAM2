import os
import json
import shutil
from collections import defaultdict

# Rutas de entrada y salida
ANNOTATIONS_PATH = "/home/ptp/sam2/datasets/severstal/total_dataset/ann"
IMAGES_PATH = "/home/ptp/sam2/datasets/severstal/total_dataset/img"
OUTPUT_BASE = "/home/ptp/sam2/datasets/severstal/defectos"

os.makedirs(OUTPUT_BASE, exist_ok=True)
class_counts = defaultdict(int)
images_per_class = defaultdict(list)
multi_defect_images = []

procesadas = 0

# Procesar cada anotación
for filename in os.listdir(ANNOTATIONS_PATH):
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(ANNOTATIONS_PATH, filename)
    with open(json_path, "r") as f:
        data = json.load(f)

    image_name = filename.replace(".jpg.json", ".jpg")
    image_path = os.path.join(IMAGES_PATH, image_name)
    if not os.path.exists(image_path):
        print(f" Imagen no encontrada: {image_name}")
        continue

    found_classes = set()
    for obj in data.get("objects", []):
        label = obj.get("classTitle")
        if label:
            found_classes.add(label)

    if not found_classes:
        print(f"[SIN CLASE] {filename}")
        continue

    if len(found_classes) == 1:
        label = list(found_classes)[0]
        class_counts[label] += 1
        images_per_class[label].append((image_path, json_path))
        print(f"[CLASE ÚNICA] {filename} → {label}")
    else:
        multi_defect_images.append((image_path, json_path))
        print(f"[MULTICLASE] {filename} → {found_classes}")

    procesadas += 1

# Crear carpetas por clase única
for defect_class, files in images_per_class.items():
    class_dir = os.path.join(OUTPUT_BASE, defect_class)
    os.makedirs(os.path.join(class_dir, "img"), exist_ok=True)
    os.makedirs(os.path.join(class_dir, "ann"), exist_ok=True)

    for img_path, json_path in files:
        shutil.copy2(img_path, os.path.join(class_dir, "img", os.path.basename(img_path)))
        shutil.copy2(json_path, os.path.join(class_dir, "ann", os.path.basename(json_path)))

# Crear carpeta multi_defect
multi_dir = os.path.join(OUTPUT_BASE, "multi_defect")
os.makedirs(os.path.join(multi_dir, "img"), exist_ok=True)
os.makedirs(os.path.join(multi_dir, "ann"), exist_ok=True)

for img_path, json_path in multi_defect_images:
    shutil.copy2(img_path, os.path.join(multi_dir, "img", os.path.basename(img_path)))
    shutil.copy2(json_path, os.path.join(multi_dir, "ann", os.path.basename(json_path)))

# Mostrar resumen
print("\n Resumen final:")
print(f"Total anotaciones procesadas: {procesadas}")
print("Conteo de imágenes por clase única:")
for label, count in class_counts.items():
    print(f" - {label}: {count}")
print(f"Imágenes con múltiples clases de defecto: {len(multi_defect_images)}")


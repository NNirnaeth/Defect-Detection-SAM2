import os
import shutil
import random

# Rutas originales de imágenes y anotaciones
IMG_DIR = "/home/ptp/sam2/datasets/severstal/train_split/img"
ANN_DIR = "/home/ptp/sam2/datasets/severstal/train_split/ann"

# Nuevas rutas de salida
BASE_OUT = "/home/ptp/sam2/datasets/severstal"
TRAIN_OUT = os.path.join(BASE_OUT, "train_inner_split")
VAL_OUT = os.path.join(BASE_OUT, "val_inner_split")

# Crear subcarpetas de salida
for path in [TRAIN_OUT, VAL_OUT]:
    os.makedirs(os.path.join(path, "img"), exist_ok=True)
    os.makedirs(os.path.join(path, "ann"), exist_ok=True)

# Listar todos los archivos JSON de anotaciones
json_files = [f for f in os.listdir(ANN_DIR) if f.endswith(".json")]
random.seed(42)
random.shuffle(json_files)

# Hacer split 90% train / 10% val
split_idx = int(0.9 * len(json_files))
train_jsons = json_files[:split_idx]
val_jsons = json_files[split_idx:]

# Función para copiar imágenes y anotaciones asociadas
def copy_pairs(json_list, target_folder):
    for json_name in json_list:
        img_name = os.path.splitext(json_name)[0]
        shutil.copy2(os.path.join(ANN_DIR, json_name), os.path.join(target_folder, "ann", json_name))
        shutil.copy2(os.path.join(IMG_DIR, img_name), os.path.join(target_folder, "img", img_name))

# Ejecutar la copia
copy_pairs(train_jsons, TRAIN_OUT)
copy_pairs(val_jsons, VAL_OUT)

# Resumen por consola
print(f" Total imágenes: {len(json_files)}")
print(f" Entrenamiento (90%): {len(train_jsons)}")
print(f" Validación (10%): {len(val_jsons)}")


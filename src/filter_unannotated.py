import os
import shutil
import json

# Directory
base_ann = "datasets/severstal/train/ann"
base_img = "datasets/severstal/train/img"
val_ann = "datasets/severstal/val_unlabeled/ann"
val_img = "datasets/severstal/val_unlabeled/img"

os.makedirs(val_ann, exist_ok=True)
os.makedirs(val_img, exist_ok=True)

moved = 0
for fname in os.listdir(base_ann):
    if not fname.endswith(".json"):
        continue

    ann_path = os.path.join(base_ann, fname)
    img_name = fname.replace(".json", "")
    img_path = os.path.join(base_img, img_name)

    # Check if file exists
    if not os.path.exists(img_path):
        continue

    # Load JSON and verify if it has bitmap annotation
    with open(ann_path, "r") as f:
        data = json.load(f)

    has_bitmap = any(obj.get("geometryType") == "bitmap" for obj in data.get("objects", []))

    if not has_bitmap:
        shutil.move(ann_path, os.path.join(val_ann, fname))
        shutil.move(img_path, os.path.join(val_img, img_name))
        moved += 1

print(f" Moved images without valid annotations: {moved}")

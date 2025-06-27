import os
import json
from sklearn.model_selection import train_test_split

## Prepare the Severstal dataset with bitmap segmentations
def prepare_dataset(data_dir="datasets/severstal/train"):
    """
    Loads the Severstal dataset with structure:
    - images:   data_dir/img/*.jpg
    - masks:    data_dir/ann/*.json (bitmap-encoded masks)

    Returns:
        train_data: list of dicts with "image" and "annotation" paths
        test_data:  same format (split 80/20)
    """
    img_dir = os.path.join(data_dir, "img")
    ann_dir = os.path.join(data_dir, "ann")

    img_files = sorted(os.listdir(img_dir))
    json_files = sorted(os.listdir(ann_dir))

    paired = []
    for img_name in img_files:
        json_name = img_name + ".json"
        json_path = os.path.join(ann_dir, json_name)
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            data = json.load(f)
            if "objects" in data and len(data["objects"]) > 0:
                paired.append({"image": img_path, "annotation": json_path})

    train_data, test_data = train_test_split(paired, test_size=0.2, random_state=42)

    print(f"Severstal dataset loaded -> Train: {len(train_data)} | Test: {len(test_data)}")
    return train_data, test_data

def prepare_severstal200(data_dir="datasets/severstal/train_200"):

    """
    Load 200 image-mask pairs for small-scale training/debugging.
    """
    img_dir = os.path.join(data_dir, "img")
    ann_dir = os.path.join(data_dir, "ann")

    img_files = sorted(os.listdir(img_dir))
    json_files = sorted(os.listdir(ann_dir))

    paired = []
    for img_name in img_files:
        ann_name = img_name + ".json"
        ann_path = os.path.join(ann_dir, ann_name)
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(ann_path):
            continue

        with open(ann_path) as f:
            data = json.load(f)
            if "objects" in data and any(obj["geometryType"] == "bitmap" for obj in data["objects"]):
                paired.append({"image": img_path, "annotation": ann_path})

    train_data, test_data = train_test_split(paired, test_size=0.2, random_state=42)
    print(f" Dataset 'train_200' loaded -> Train: {len(train_data)} | Test: {len(test_data)}")
    return train_data, test_data

def prepare_subset(name):
    """
    Load severstal subset like train_25, train_50, etc.
    """
    data_dir = f"datasets/severstal/{name}"
    img_dir = os.path.join(data_dir, "img")
    ann_dir = os.path.join(data_dir, "ann")

    paired = []
    for fname in sorted(os.listdir(img_dir)):
        ann_path = os.path.join(ann_dir, fname + ".json")
        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(ann_path):
            continue
        with open(ann_path) as f:
            data = json.load(f)
            if "objects" in data and any(obj.get("geometryType") == "bitmap" for obj in data["objects"]):
                paired.append({"image": img_path, "annotation": ann_path})

    train_data, test_data = train_test_split(paired, test_size=0.2, random_state=42)
    print(f"Subset '{name}' loaded -> Train: {len(train_data)} | Test: {len(test_data)}")
    return train_data, test_data

def prepare_custom_split(train_dir, test_dir):
    def load_pairs(directory):
        img_dir = os.path.join(directory, "img")
        ann_dir = os.path.join(directory, "ann")
        paired = []

        for fname in sorted(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, fname)
            ann_path = os.path.join(ann_dir, fname + ".json")
            if not os.path.exists(ann_path):
                continue
            with open(ann_path) as f:
                data = json.load(f)
                if "objects" in data and any(obj.get("geometryType") == "bitmap" for obj in data["objects"]):
                    paired.append({"image": img_path, "annotation": ann_path})
        return paired

    train_data = load_pairs(train_dir)
    test_data = load_pairs(test_dir)

    print(f" Custom Severstal split loaded -> Train: {len(train_data)} | Test: {len(test_data)}")
    return train_data, test_data

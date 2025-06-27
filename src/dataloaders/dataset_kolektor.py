import os
import glob
import json
from sklearn.model_selection import train_test_split

## Prepare the KolektorSDD2 dataset with JSON RLE annotations
def prepare_dataset(data_dir="datasets/kolektorSDD2"):
    """
    Prepares the KolektorSDD2 dataset assuming the following structure:
    - data_dir/train/*.jpg + *.json
    - data_dir/test/*.jpg + *.json
    - data_dir/val/*.jpg + *.json

    Returns:
        train_data: list of dicts (image + annotations)
        test_data: same format
    """
    subsets = ["train", "test", "val"]
    data_splits = {subset: [] for subset in subsets}

    for subset in subsets:
        subset_dir = os.path.join(data_dir, subset)
        image_files = glob.glob(os.path.join(subset_dir, "*.jpg"))
        json_files = glob.glob(os.path.join(subset_dir, "*.json"))

        print(f"Looking for images in {subset_dir}: {len(image_files)} found")
        print(f"Looking for annotations in {subset_dir}: {len(json_files)} found")

        for image_path in image_files:
            image_name = os.path.basename(image_path)
            json_path = os.path.join(subset_dir, image_name.replace(".jpg", ".json"))
            if not os.path.exists(json_path):
                print(f"⚠️ Warning: missing annotation for {image_path}")
                continue

            # Load and check annotation content
            with open(json_path, "r") as f:
                ann_data = json.load(f)

            if "annotations" not in ann_data or len(ann_data["annotations"]) == 0:
                print(f"⏭️ Skipping: no valid annotations in {json_path}")
                continue

            data_splits[subset].append(ann_data)

    train_data = data_splits["train"]
    test_data = data_splits["test"]

    print(f"Dataset 'kolektor' -> Train: {len(train_data)}, Test: {len(test_data)}")
    return train_data, test_data


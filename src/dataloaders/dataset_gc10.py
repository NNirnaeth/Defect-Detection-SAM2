import os
import glob
import json
from sklearn.model_selection import train_test_split

## Prepare the GC10-DET dataset: returns training and test samples with image/mask pairs
def prepare_dataset(data_dir="datasets/gc10-DET", test_size=0.2):
    """
    Loads GC10-DET dataset with structure:
    - images in: data_dir/ds/img/
    - annotations in: data_dir/ds/ann/*.json

    Returns:
        train_data: list of {"image": image_path, "annotation": mask_path}
        test_data:  list of {"image": image_path, "annotation": mask_path}
    """
    img_dir = os.path.join(data_dir, "ds", "img")
    ann_dir = os.path.join(data_dir, "ds", "ann")

    # Match images to corresponding annotation files
    image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    data = []
    for img_path in image_files:
        image_name = os.path.basename(img_path)
        json_name = image_name + ".json"
        json_path = os.path.join(ann_dir, json_name)
        if os.path.exists(json_path):
            data.append({"image": img_path, "annotation": json_path})
        else:
            print(f"Warning: annotation not found for {image_name}, skipping.")

    # Split into train and test
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    print(f"GC10-DET dataset loaded -> Train: {len(train_data)} | Test: {len(test_data)}")
    return train_data, test_data


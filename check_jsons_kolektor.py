import os
import json
from pycocotools import mask as coco_mask

# Set the folder to check: train, test or val
folder = "datasets/kolektorSDD2/train"
json_files = [f for f in os.listdir(folder) if f.endswith(".json")]

total_files = 0
with_annotations = 0
with_segmentation = 0
decodable_masks = 0
decode_errors = 0

for file in json_files:
    json_path = os.path.join(folder, file)
    total_files += 1


    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        annotations = data.get("annotations", [])
        if len(annotations) == 0:
            continue

        with_annotations += 1

        for ann in annotations:
            if "segmentation" in ann:
                with_segmentation += 1
                rle = ann["segmentation"]

                try:
                    rle_obj = coco_mask.frPyObjects(rle, rle["size"][0], rle["size"][1])
                    _ = coco_mask.decode(rle_obj)
                    decodable_masks += 1
                except Exception as e:
                    print(f" Decode error in {file}: {e}")
                    decode_errors += 1

    except Exception as e:
        print(f"File read error in {file}: {e}")
        decode_errors += 1

# Summary

print("\n SUMMARY")
print(f"Total JSON files: {total_files}")
print(f"With non-empty annotations: {with_annotations}")
print(f"With segmentation data: {with_segmentation}")
print(f"Successfully decodable masks: {decodable_masks}")
print(f"Decode errors: {decode_errors}")



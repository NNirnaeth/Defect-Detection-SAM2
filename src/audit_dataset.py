import os
import json
import cv2
import base64
import zlib
import numpy as np

def decode_bitmap(data):
    """
    Decode base64 PNG or zlib-compressed bitmap used in Severstal annotations.
    Returns a binary mask (numpy array) or None if decoding fails.
    """
    if not data:
        return None

    try:
        decoded_data = base64.b64decode(data)

        try:
            decompressed_data = zlib.decompress(decoded_data)
        except zlib.error:
            decompressed_data = decoded_data  # Assume raw PNG

        # Detect PNG by signature
        if decompressed_data[:8] == b'\x89PNG\r\n\x1a\n':
            nparr = np.frombuffer(decompressed_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

            if img is None:
                return None

            if len(img.shape) == 3:
                # RGBA or RGB
                if img.shape[2] == 4:
                    mask = img[:, :, 3]  # Alpha channel
                else:
                    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                mask = img  # Already grayscale

            return (mask > 0).astype(np.uint8)

        # Fallback for raw bitmap headers
        if len(decompressed_data) >= 8:
            width = int.from_bytes(decompressed_data[0:4], byteorder='little')
            height = int.from_bytes(decompressed_data[4:8], byteorder='little')
            bitmap_data = decompressed_data[8:]
            if len(bitmap_data) >= width * height:
                mask = np.frombuffer(bitmap_data, dtype=np.uint8, count=width * height)
                return mask.reshape((height, width))

    except Exception as e:
        print(f"[decode_bitmap] Error: {e}")
        return None

def audit_folder(image_dir, ann_dir):
    total = 0
    matched = 0
    errors = 0

    for fname in os.listdir(image_dir):
        if not fname.endswith(".jpg"):
            continue

        image_path = os.path.join(image_dir, fname)
        json_path = os.path.join(ann_dir, fname + ".json")

        total += 1

        if not os.path.exists(json_path):
            errors += 1
            continue

        try:
            with open(json_path) as f:
                data = json.load(f)
            H, W = data["size"]["height"], data["size"]["width"]
            full_mask = np.zeros((H, W), dtype=np.uint8)

            for obj in data.get("objects", []):
                if obj.get("geometryType") == "bitmap" and "bitmap" in obj:
                    mask = decode_bitmap(obj["bitmap"]["data"])
                    if mask is not None:
                        matched += 1
        except:
            errors += 1

    return total, matched, errors

def audit_all_subsets(base_path="/home/ptp/sam2/datasets/severstal"):
    subsets = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith(("train", "val", "test"))]

    print(f"{'Subset':<15} | {'Total':<6} | {'Decoded':<8} | {'Errors':<6}")
    print("-" * 45)
    for subset in subsets:
        img_dir = os.path.join(base_path, subset, "img")
        ann_dir = os.path.join(base_path, subset, "ann")
        if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
            continue
        total, matched, errors = audit_folder(img_dir, ann_dir)
        print(f"{subset:<15} | {total:<6} | {matched:<8} | {errors:<6}")

if __name__ == "__main__":
    audit_all_subsets()


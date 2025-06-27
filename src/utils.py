"""
Utility functions for data loading, mask processing, and evaluation metrics.
"""

import os
import cv2
import numpy as np
from pycocotools import mask as coco_mask
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import base64
import zlib

## Draw binary mask from rectangle-based annotation (Roboflow style)
def draw_rectangle_mask_from_json(json_data):
    h, w = json_data["size"]["height"], json_data["size"]["width"]
    mask = np.zeros((h, w), dtype=np.uint8)
    for obj in json_data.get("objects", []):
        if obj.get("geometryType") == "rectangle":
            points = obj["points"]["exterior"]
            pt1 = tuple(points[0])  # top-left
            pt2 = tuple(points[1])  # bottom-right
            cv2.rectangle(mask, pt1, pt2, color=1, thickness=-1)
    return mask


## Decode Severstal bitmap mask PNG from JSON

def decode_bitmap_to_mask(data):
    """
    Decode base64 data, decompress it using zlib, and handle PNG image data.
    Returns a numpy array representing the mask.
    """
    if not data:
        print("No data provided for bitmap decoding")
        return None

    try:
        # Step 1: Decode base64 data
        decoded_data = base64.b64decode(data)

        # Step 2: Decompress the data using zlib
        try:
            decompressed_data = zlib.decompress(decoded_data)
        except zlib.error:
            # If decompression fails, try direct loading - might be raw PNG
            print("Zlib decompression failed, trying direct loading...")
            decompressed_data = decoded_data

        # Step 3: Check for PNG signature
        png_signature = b'\x89PNG\r\n\x1a\n'
        if decompressed_data[:8] == png_signature:
            # This is a PNG file, load it directly with cv2
            try:
                # Convert PNG data to numpy array
                nparr = np.frombuffer(decompressed_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

                if img is None:
                    print("CV2 failed to decode the PNG data")
                    return None

                # Handle different channel configurations
                if len(img.shape) == 3 and img.shape[2] >= 3:
                    # Convert multi-channel masks to single-channel
                    # Use the first channel, or the alpha channel if it exists
                    if img.shape[2] == 4:  # RGBA
                        mask = img[:, :, 3]  # Use alpha channel
                    else:
                        # Convert to grayscale
                        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    # Already a grayscale image
                    mask = img

                # Normalize mask: any non-zero pixel becomes 1
                mask = (mask > 0).astype(np.uint8) * 255
                return mask
            except Exception as e:
                print(f"Error loading PNG with cv2: {e}")
                return None
        else:
            # Not a PNG, try to interpret as raw bitmap data
            try:
                # Check if the data starts with width/height values
                if len(decompressed_data) >= 8:
                    width = int.from_bytes(decompressed_data[0:4], byteorder='little')
                    height = int.from_bytes(decompressed_data[4:8], byteorder='little')

                    # Sanity check for reasonable dimensions
                    if 0 < width < 10000 and 0 < height < 10000:
                        print(f"Decoded bitmap dimensions: {width} x {height}")
                        # Skip the 8-byte header
                        bitmap_data = decompressed_data[8:]

                        # Create mask from bitmap data
                        # Assuming 1 byte per pixel in row-major order
                        if len(bitmap_data) >= width * height:
                            mask = np.frombuffer(bitmap_data, dtype=np.uint8, count=width * height)
                            mask = mask.reshape(height, width)
                            return mask

                # If we get here, we couldn't parse using standard methods
                print("Could not interpret bitmap data format")
                return None
            except Exception as e:
                print(f"Error parsing bitmap data: {e}")
                return None

    except Exception as e:
        print(f"Error in decode_png_bitmap: {e}")
        return None

## Read and preprocess a single image and its binary mask (PNG, RLE or bitmap JSON)
def read_image(image_path, mask_path):

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Error: Image not found at {image_path}")
    img = img[..., ::-1]  # Convert BGR to RGB

    # Load mask depending on format
    if mask_path.endswith((".png", ".jpg", ".jpeg")):
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise FileNotFoundError(f"Error: Mask not found at {mask_path}")

    elif mask_path.endswith(".json"):
        with open(mask_path, "r") as f:
            mask_data = json.load(f)

        if "annotations" in mask_data:
            h, w = mask_data["image"]["height"], mask_data["image"]["width"]
            mask = np.zeros((h, w), dtype=np.uint8)
            for ann in mask_data["annotations"]:
                if "segmentation" in ann:
                    rle = ann["segmentation"]
                    decoded = coco_mask.decode(rle).astype(np.uint8)
                    mask = np.maximum(mask, decoded)

        elif "objects" in mask_data:
            h, w = mask_data["size"]["height"], mask_data["size"]["width"]
            mask = np.zeros((h, w), dtype=np.uint8)

            for obj in mask_data["objects"]:
                if obj.get("geometryType") == "bitmap":
                    bitmap_data = obj["bitmap"]["data"]
                    origin_x, origin_y = obj["bitmap"]["origin"]
                    sub_mask = decode_bitmap_to_mask(bitmap_data)
                    if sub_mask is None:
                        continue

                    end_y = origin_y + sub_mask.shape[0]
                    end_x = origin_x + sub_mask.shape[1]

                    if end_y > h or end_x > w:
                        continue

                    mask[origin_y:end_y, origin_x:end_x] = np.maximum(
                        mask[origin_y:end_y, origin_x:end_x], sub_mask
                    )

                elif obj.get("geometryType") == "rectangle":
                    mask = np.maximum(mask, draw_rectangle_mask_from_json(mask_data))
        else:
            raise ValueError(f"Unknown JSON annotation format: {mask_path}")

    else:
        raise ValueError(f"Unsupported mask format: {mask_path}")

    # Resize image and mask to max 1024px
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return img, mask


## Unified batch reader for image or RLE annotations

def read_batch(data, base_path='', annotation_format='image', visualize_data=False):
    ent = random.choice(data)
    img_path = os.path.join(base_path, ent["image"] if isinstance(ent["image"], str) else ent["image"]["file_name"])
    Img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if Img is None:
        print(f"Error reading image {img_path}")
        return None, None, None, 0

    if annotation_format == 'rle':
        h, w = ent["image"]["height"], ent["image"]["width"]
        ann_map = np.zeros((h, w), dtype=np.uint8)
        for ann in ent["annotations"]:
            rle = ann["segmentation"]
            decoded = coco_mask.decode(rle).astype(np.uint8)
            ann_map = np.maximum(ann_map, decoded)

    elif annotation_format == 'bitmap':
        try:
            with open(ent["annotation"], "r") as f:
                json_data = json.load(f)

            height = json_data["size"]["height"]
            width = json_data["size"]["width"]
            ann_map = np.zeros((height, width), dtype=np.uint8)

            for obj in json_data.get("objects", []):
                if obj.get("geometryType") != "bitmap":
                    continue
                bitmap_data = obj["bitmap"]["data"]
                origin_x, origin_y = obj["bitmap"]["origin"]

                sub_mask = decode_bitmap_to_mask(bitmap_data)
                if sub_mask is None:
                    continue

                end_y = origin_y + sub_mask.shape[0]
                end_x = origin_x + sub_mask.shape[1]

                if end_y > height or end_x > width:
                    continue

                ann_map[origin_y:end_y, origin_x:end_x] = np.maximum(
                    ann_map[origin_y:end_y, origin_x:end_x], sub_mask)

        except Exception as e:
            print(f"[Error decoding bitmap] {ent['annotation']}: {e}")
            return None, None, None, 0

    else:
        ann_path = os.path.join(base_path, ent["annotation"])
        ann_map = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        if ann_map is None:
            print(f"Error reading mask {ann_path}")
            return None, None, None, 0

    Img = Img[..., ::-1]  # BGR to RGB
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])
    img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    binary_mask = (ann_map > 0).astype(np.uint8)
    eroded = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)

    coords = np.argwhere(eroded > 0)
    np.random.shuffle(coords)
    points = coords[:10]
    points = np.expand_dims(points, axis=1) if len(points) > 0 else np.zeros((0, 1, 2))

    binary_mask = np.expand_dims(binary_mask, axis=0)

    if visualize_data:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title('Image')
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Binary Mask')
        plt.imshow(binary_mask[0], cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Mask + Points')
        plt.imshow(binary_mask[0], cmap='gray')
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, point in enumerate(points):
            plt.scatter(point[0][0], point[0][1], c=colors[i % len(colors)], s=100)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    n_instances = len(np.unique(ann_map)) - 1  # ignore background 0
    return img, binary_mask, points, n_instances


## Decode RLE into binary mask
def decode_rle_to_mask(rle):
    return coco_mask.decode(rle).astype(np.uint8)

## Compute IoU between predicted and ground truth masks
def compute_iou(pred, gt):
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / float(union + 1e-6)

## Sample random points from inside a binary mask
def get_points(mask, num_points):
    coords = np.argwhere(mask > 0)
    points = []
    for _ in range(num_points):
        yx = coords[np.random.randint(len(coords))]
        points.append([[yx[1], yx[0]]])
    return np.array(points)

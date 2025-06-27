import os
import random
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pycocotools import mask as coco_mask
import json

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils import read_image, get_points
from metrics_logger import MetricsLogger
from utils import decode_bitmap_to_mask



## Evaluate a single image and visualize the result
def eval_one_image(args, test_data, num_samples=30):
    # Load SAM2 model and predictor
    sam2_model = build_sam2(args.config, args.checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    # Load fine-tuned weights
    fine_tuned_checkpoint = torch.load(args.fine_tuned_checkpoint, map_location="cuda")
    predictor.model.load_state_dict(fine_tuned_checkpoint)

    # Randomly select a test image and load corresponding mask
    selected_entry = random.choice(test_data)
    image_path = selected_entry['image']
    mask_path = selected_entry['annotation']
    image, mask = read_image(image_path, mask_path)

    # Generate random point prompts within the mask
    input_points = get_points(mask, num_samples)

    # Perform inference using the point prompts
    with torch.no_grad():
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )

    # Sort predicted masks by confidence scores
    np_masks = np.array(masks[:, 0])
    np_scores = scores[:, 0]
    sorted_masks = np_masks[np.argsort(np_scores)][::-1]

    # Merge non-overlapping masks into a final segmentation map
    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
    occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)
    for i in range(sorted_masks.shape[0]):
        mask = sorted_masks[i]
        if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
            continue
        mask_bool = mask.astype(bool)
        mask_bool[occupancy_mask] = False
        seg_map[mask_bool] = i + 1
        occupancy_mask[mask_bool] = True

    # Plot original image, ground truth mask, and final segmentation result
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title('Test Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Original Mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Final Segmentation')
    plt.imshow(seg_map, cmap='jet')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

## Decode RLE mask into binary mask
def decode_rle_to_mask(rle):
    return coco_mask.decode(rle).astype(np.uint8)

## Decode bitmap annotations from JSON entry
def decode_bitmap_from_entry(entry):
    h = entry["size"]["height"]
    w = entry["size"]["width"]
    mask = np.zeros((h, w), dtype=np.uint8)

    for obj in entry["objects"]:
        if obj.get("geometryType") != "bitmap":
            continue
        origin_x, origin_y = obj["bitmap"]["origin"]
        sub_mask = decode_bitmap_to_mask(obj["bitmap"]["data"])
        if sub_mask is None:
            continue
        end_y = origin_y + sub_mask.shape[0]
        end_x = origin_x + sub_mask.shape[1]
        if end_y > h or end_x > w:
            continue
        mask[origin_y:end_y, origin_x:end_x] = np.maximum(
            mask[origin_y:end_y, origin_x:end_x], sub_mask
        )
    return mask

## Merge all masks of an image into one binary ground truth mask
def get_binary_gt_mask(entry, base_path=''):
    with open(entry["annotation"], "r") as f:
        data = json.load(f)

    if "annotations" in data:
        h, w = data["image"]["height"], data["image"]["width"]
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in data["annotations"]:
            decoded = decode_rle_to_mask(ann["segmentation"])
            gt_mask = np.maximum(gt_mask, decoded)
        return gt_mask
    elif "objects" in data:
        return decode_bitmap_from_entry(data)
    else:
        raise ValueError("Entry must contain either 'annotations' or 'objects'")




## Sample random points from inside a binary mask
def sample_points_from_mask(mask, num_points=30):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    chosen_idx = np.random.choice(len(coords), min(num_points, len(coords)), replace=False)
    return coords[chosen_idx]

## Compute Intersection over Union (IoU) between predicted and GT masks
def compute_iou(pred, gt):
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / float(union + 1e-6)



## Evaluate the full test set and report average IoU and thresholds

def eval_full_test(args, test_data, base_path='', iou_thresholds=[0.5, 0.75, 0.95]):
    # Load SAM2 model and predictor
    sam2_model = build_sam2(args.config, args.checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    # Load fine-tuned weights
    ft_ckpt = torch.load(args.fine_tuned_checkpoint, map_location="cuda")
    predictor.model.load_state_dict(ft_ckpt)

    all_ious = []
    gt_masks_list = []
    pred_masks_list = []

    for entry in test_data:
        img_file = entry["image"]
        mask_file = entry["annotation"]

        if not os.path.exists(img_file):
            print(f"Path does not exist: {img_file}")
            continue

        Img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        if Img is None:
            print(f"Image load failed: {img_file}")
            continue
        Img = Img[..., ::-1]

        try:
            gt_mask = get_binary_gt_mask(entry, base_path)
        except Exception as ex:
            print(f"Cannot get GT mask: {entry}: {ex}")
            continue

        points = sample_points_from_mask(gt_mask, num_points=30)
        if points is None:
            continue

        with torch.no_grad():
            predictor.set_image(Img.copy())
            masks, scores, _ = predictor.predict(
                point_coords=points,
                point_labels=np.ones((points.shape[0],), dtype=np.int32)
            )

        np_masks = np.array(masks[:, 0], dtype=np.uint8)
        np_scores = scores.flatten()
        sorted_idxs = np.argsort(np_scores)[::-1]
        sorted_masks = np_masks[sorted_idxs]

        seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
        occupancy = np.zeros_like(seg_map, dtype=bool)
        for i, msk in enumerate(sorted_masks):
            m_bool = msk.astype(bool)
            if (m_bool & occupancy).sum() / (msk.sum() + 1e-6) > 0.15:
                continue
            m_bool[occupancy] = False
            seg_map[m_bool] = i + 1
            occupancy[m_bool] = True

        pred_mask = (seg_map > 0).astype(np.uint8)
        iou_val = compute_iou(pred_mask, (gt_mask > 0).astype(np.uint8))
        all_ious.append(iou_val)

        gt_masks_list.append((gt_mask > 0).astype(np.uint8))
        pred_masks_list.append(pred_mask)

    all_ious = np.array(all_ious)
    results = {}
    for th in iou_thresholds:
        results[f'IoU@{int(th * 100)}'] = np.mean(all_ious >= th)

    print("\nSAM2 Evaluation on test set:")
    print(f"Average IoU: {all_ious.mean():.4f}")
    for k, v in results.items():
        print(f"{k}: {v * 100:.2f}%")

    # Métricas extendidas
    extended = evaluate_metrics_extended(predictions=pred_masks_list, ground_truths=gt_masks_list)

    # Save basic metrics
    logger = MetricsLogger("logs/eval_metrics.csv")
    logger.log(
        mode="eval",
        dataset=args.dataset,
        steps=args.steps if hasattr(args, 'steps') else "N/A",
        model_name=os.path.basename(args.fine_tuned_checkpoint),
        iou=all_ious.mean(),
        iou50=results.get("IoU@50", 0),
        iou75=results.get("IoU@75", 0),
        iou95=results.get("IoU@95", 0)
    )

    return results


def evaluate_on_validation_set(predictor, val_data, args):
    all_ious = []
    for entry in val_data:
        img_file = entry["image"]
        ann_file = entry["annotation"]

        if not os.path.exists(img_file):
            continue
        Img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        if Img is None:
            continue
        Img = Img[..., ::-1]  # BGR a RGB

        try:
            gt_mask = get_binary_gt_mask(entry)
        except Exception:
            continue

        points = sample_points_from_mask(gt_mask, num_points=30)
        if points is None:
            continue

        with torch.no_grad():
            predictor.set_image(Img.copy())
            masks, scores, _ = predictor.predict(
                point_coords=points,
                point_labels=np.ones((points.shape[0],), dtype=np.int32)
            )

        np_masks = np.array(masks[:, 0], dtype=np.uint8)
        np_scores = scores.flatten()
        sorted_idxs = np.argsort(np_scores)[::-1]
        sorted_masks = np_masks[sorted_idxs]

        seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
        occupancy = np.zeros_like(seg_map, dtype=bool)
        for i, msk in enumerate(sorted_masks):
            m_bool = msk.astype(bool)
            if (m_bool & occupancy).sum() / (msk.sum() + 1e-6) > 0.15:
                continue
            m_bool[occupancy] = False
            seg_map[m_bool] = i + 1
            occupancy[m_bool] = True

        pred_mask = (seg_map > 0).astype(np.uint8)
        iou_val = compute_iou(pred_mask, (gt_mask > 0).astype(np.uint8))
        all_ious.append(iou_val)

    if len(all_ious) == 0:
        return 0.0

    return float(np.mean(all_ious))

def evaluate_metrics_extended(predictions, ground_truths, thresholds=[0.5]):
    """
    Calcula métricas: IoU medio, IoU@50, Precision, Recall, F1 y benevolente.
    predictions y ground_truths deben ser listas de binarios 2D del mismo tamaño.
    """
    ious = []
    tp = fp = fn = 0
    benevolente_hits = 0

    for pred, gt in zip(predictions, ground_truths):
        pred = (pred > 0).astype(np.uint8)
        gt = (gt > 0).astype(np.uint8)

        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        iou = inter / (union + 1e-6)
        ious.append(iou)

        if iou > 0:
            if iou >= 0.5:
                tp += 1
            else:
                fp += 1
                fn += 1  # penaliza como falso negativo también

            # Benevolente: pred cubre ≥75% del GT
            if (inter / (gt.sum() + 1e-6)) >= 0.75:
                benevolente_hits += 1
        else:
            if pred.sum() > 0:
                fp += 1
            if gt.sum() > 0:
                fn += 1

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print("\n--- Métricas extendidas ---")
    print(f"Average IoU: {np.mean(ious):.4f}")
    print(f"IoU@50: {(np.array(ious) >= 0.5).mean() * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Benevolente (area pred cubre ≥75% GT): {benevolente_hits / len(ground_truths) * 100:.2f}%")

    return {
        "IoU_mean": np.mean(ious),
        "IoU@50": (np.array(ious) >= 0.5).mean(),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Benevolente": benevolente_hits / len(ground_truths)
    }

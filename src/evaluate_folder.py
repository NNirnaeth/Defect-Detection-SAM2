"""
Script to evaluate SAM2 model on a folder of images and save visual results.
This script:
1. Loads SAM2 model and fine-tuned weights
2. Processes each image in the input folder (limited to 100 images)
3. Automatically finds all masks using SAM2's automatic mask generator
4. Saves results with original image, masks, and overlay
5. Measures and reports inference times in real-time
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from statistics import median
from pathlib import Path
import yaml
import random

# Add the libs directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs/sam2base"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def load_config(config_path):
    """Load model configuration from yaml file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_debug_masks(image, masks, output_dir, filename):
    """Save individual masks for debugging"""
    masks_dir = os.path.join(output_dir, "debug_masks", filename)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Save original image for reference
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(os.path.join(masks_dir, "original.jpg"), bbox_inches='tight')
    plt.close()
    
    # Save each mask separately
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        score = mask_data['predicted_iou']
        area = mask_data['area']
        area_ratio = area / (image.shape[0] * image.shape[1])
        
        # Create a visualization
        plt.figure(figsize=(15, 5))
        
        # Original with mask overlay
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5, cmap='jet')
        plt.title(f"Mask {i+1} Overlay")
        plt.axis('off')
        
        # Mask only
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Mask {i+1}")
        plt.axis('off')
        
        # Add text with mask stats
        plt.subplot(1, 3, 3)
        plt.text(0.1, 0.6, f"Score: {score:.4f}", fontsize=12)
        plt.text(0.1, 0.5, f"Area: {area} pixels", fontsize=12)
        plt.text(0.1, 0.4, f"Area ratio: {area_ratio:.4f}", fontsize=12)
        plt.text(0.1, 0.3, f"Shape: {mask.shape}", fontsize=12)
        plt.axis('off')
        
        plt.savefig(os.path.join(masks_dir, f"mask_{i+1}_score_{score:.4f}.jpg"), bbox_inches='tight')
        plt.close()

def save_compact_results(image, pred_mask, scores, output_dir, filename, mask_count):
    """
    Save results in a compact format with 3 rows:
    1. Original image
    2. Predicted masks with scores
    3. Overlay (50% alpha)
    
    Args:
        image: Original RGB image
        pred_mask: Segmentation mask with different values for different objects
        scores: Confidence scores for each mask
        output_dir: Directory to save results
        filename: Base filename for output
        mask_count: Number of masks found
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 3 rows
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f'Original Image - Found {mask_count} masks')
    axes[0].axis('off')
    
    # Check if we have any masks
    unique_segments = np.unique(pred_mask)
    if len(unique_segments) <= 1:
        print(f"WARNING: No masks found in image {filename}!")
    
    # Use a colormap with distinct colors for each class
    # 'tab20' provides 20 distinct colors and is better for class visualization than 'jet'
    cmap = plt.cm.get_cmap('tab20', len(unique_segments) if len(unique_segments) > 1 else 2)
    
    # Create a colored mask image where each class has a unique color
    colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 4), dtype=np.float32)
    
    for i, segment_id in enumerate(unique_segments):
        if segment_id == 0:  # Skip background
            continue
        # Get mask for this segment
        mask = pred_mask == segment_id
        # Assign a unique color from the colormap
        color = cmap(i % cmap.N)
        # Set the color for this segment in the colored mask
        colored_mask[mask] = color
    
    # Predicted mask
    axes[1].imshow(colored_mask)
    axes[1].set_title(f'Predicted Masks: {mask_count}')
    axes[1].axis('off')
    
    # Add colorbar with scores if we have unique segments
    if len(unique_segments) > 1:
        # Create a separate axis for the colorbar
        cbar_ax = fig.add_axes([0.92, 0.4, 0.02, 0.2])  # [left, bottom, width, height]
        
        # Create a colorbar
        norm = plt.Normalize(vmin=1, vmax=len(unique_segments))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        
        # Add scores to colorbar if we have them
        if scores is not None and len(scores) >= len(unique_segments) - 1:
            # Skip background (0)
            segment_ids = unique_segments[unique_segments > 0]
            
            # Prepare tick positions and labels
            tick_positions = np.arange(1, len(segment_ids) + 1)
            tick_labels = [f"ID:{seg_id} ({scores[seg_id-1]:.2f})" for seg_id in segment_ids]
            
            # Set ticks on colorbar
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels)
    
    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(colored_mask)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    # Save compact result as JPG
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_result.jpg"), 
                bbox_inches='tight', pad_inches=0.1, dpi=300, format='jpg')
    plt.close()

def process_image_automatic(mask_generator, image, debug=False, output_dir=None, filename=None, save_individual=False):
    """
    Process a single image with SAM2 model using the automatic mask generator.
    
    Args:
        mask_generator: SAM2 automatic mask generator
        image: Input RGB image
        debug: Whether to save debug information
        output_dir: Output directory for debug info
        filename: Base filename for output
        save_individual: Whether to save individual masks
        
    Returns:
        seg_map: Segmentation mask with different values for different objects
        scores_list: List of confidence scores for each mask
        mask_count: Number of masks found
        inference_time: Time taken for inference
    """
    # Resize image to 1024 max dimension while preserving aspect ratio
    h, w = image.shape[:2]
    r = min(1024 / w, 1024 / h)
    resized_image = cv2.resize(image, (int(w * r), int(h * r)))
    
    # Run inference with automatic mask generator
    start_time = time.time()
    try:
        masks = mask_generator.generate(resized_image)
        inference_time = time.time() - start_time
        
        # Print mask information for debugging
        print(f"\nFound {len(masks)} masks in image")
        
        # Filter out masks that are too large (whole image masks)
        total_pixels = resized_image.shape[0] * resized_image.shape[1]
        filtered_masks = []
        for mask in masks:
            area_ratio = mask['area'] / total_pixels
            print(f"  Mask score: {mask['predicted_iou']:.4f}, area: {mask['area']} pixels, area ratio: {area_ratio:.4f}")
            # Filter out masks that are more than 90% of the image
            if area_ratio < 0.9:
                filtered_masks.append(mask)
        
        # Update masks with filtered list
        masks = filtered_masks
        print(f"  After filtering, {len(masks)} masks remain")
        
        # Create segmentation map and collect scores
        seg_map = np.zeros((resized_image.shape[0], resized_image.shape[1]), dtype=np.uint8)
        scores_list = []
        
        # Sort masks by predicted IoU score (highest first)
        masks = sorted(masks, key=lambda x: x['predicted_iou'], reverse=True)
        
        # Save debug information if requested
        if debug and output_dir and filename:
            save_debug_masks(resized_image, masks, output_dir, filename)
        
        # Save individual masks if requested
        if save_individual and output_dir and filename:
            save_individual_masks(resized_image, masks, output_dir, filename)
        
        # Populate segmentation map with masks
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            score = mask_data['predicted_iou']
            
            # Add this mask with ID (i+1)
            mask_id = i + 1
            seg_map[mask] = mask_id
            scores_list.append(score)
        
        # Resize segmentation map back to original image size
        seg_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return seg_map, scores_list, len(masks), inference_time
    
    except Exception as e:
        inference_time = time.time() - start_time
        print(f"Error during mask generation: {e}")
        print(f"Creating empty mask as fallback")
        
        # Create empty segmentation map
        seg_map = np.zeros((h, w), dtype=np.uint8)
        return seg_map, [], 0, inference_time

def update_stats(inference_times):
    """Print current inference time statistics"""
    times = np.array(inference_times)
    print("\n--- Current Statistics (seconds) ---")
    print(f"Images processed: {len(times)}")
    print(f"Min: {times.min():.4f}")
    print(f"Max: {times.max():.4f}")
    print(f"Mean: {times.mean():.4f}")
    print(f"Median: {median(times):.4f}")
    print(f"Total time: {times.sum():.2f}")
    print("-----------------------------------")

def save_individual_masks(image, masks, output_dir, filename):
    """
    Save each individual mask separately for detailed analysis.
    
    Args:
        image: Original RGB image
        masks: List of mask data dictionaries from SAM2
        output_dir: Directory to save results
        filename: Base filename for output
    """
    if not masks:
        print(f"  No masks to save for {filename}")
        return
        
    # Create directory for this image
    masks_dir = os.path.join(output_dir, "individual_masks", filename)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Save original image
    cv2.imwrite(os.path.join(masks_dir, "original.jpg"), image[..., ::-1])  # RGB to BGR for OpenCV
    
    # Create a composite image with all masks
    composite = np.zeros_like(image)
    
    # Different colors for different masks
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    
    # Save each mask individually and add to composite
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        score = mask_data['predicted_iou']
        
        # Create colored mask image (binary mask)
        mask_vis = np.zeros_like(image)
        color = colors[i % len(colors)]
        for c in range(3):
            mask_vis[:, :, c] = mask * color[c]
        
        # Save individual mask visualization
        mask_filename = f"mask_{i+1}_score_{score:.4f}.jpg"
        cv2.imwrite(os.path.join(masks_dir, mask_filename), mask_vis)
        
        # Add to composite (with alpha blending)
        alpha = 0.5
        composite = np.where(
            mask[:, :, np.newaxis], 
            composite * (1 - alpha) + mask_vis * alpha, 
            composite
        )
    
    # Save composite visualization
    cv2.imwrite(os.path.join(masks_dir, "all_masks_composite.jpg"), composite.astype(np.uint8))
    
    # Create overlay of masks on original image
    overlay = image.copy()
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        color = colors[i % len(colors)]
        for c in range(3):
            overlay[:, :, c] = np.where(mask, overlay[:, :, c] * 0.7 + color[c] * 0.3, overlay[:, :, c])
    
    # Save overlay
    cv2.imwrite(os.path.join(masks_dir, "overlay.jpg"), overlay[..., ::-1])  # RGB to BGR for OpenCV

def evaluate_folder(args):
    """
    Evaluate SAM2 model on a folder of images.
    
    Args:
        args: Command line arguments containing:
            - input_folder: Path to folder with images
            - checkpoint: Path to SAM2 checkpoint
            - config: Path to SAM2 config
            - fine_tuned_checkpoint: Path to fine-tuned weights
            - max_images: Maximum number of images to process
            - debug: Enable debug mode with verbose output
            - save_individual_masks: Whether to save individual masks
    """
    # Load config
    config = load_config(args.config)
    
    # Debug mode
    debug_mode = args.debug
    save_individual = args.save_individual_masks
    
    if debug_mode:
        print(f"DEBUG MODE ENABLED")
        print(f"Config loaded from: {args.config}")
        print(f"Config contents: {config}")
        
    if save_individual:
        print(f"INDIVIDUAL MASK SAVING ENABLED - Will save detailed visualizations")
    
    # Load model
    print(f"Loading SAM2 model from checkpoint: {args.checkpoint}")
    try:
        sam2_model = build_sam2(args.config, args.checkpoint, device="cpu")
        print("SAM2 model structure:")
        print(f"Model type: {type(sam2_model)}")
        if debug_mode:
            for name, param in sam2_model.named_parameters():
                if param.requires_grad:
                    print(f"Trainable parameter: {name} | Shape: {param.shape}")
            print("---")
    except Exception as e:
        print(f"Error loading base model: {e}")
        raise
    
    # Load fine-tuned weights
    print(f"Loading fine-tuned weights from {args.fine_tuned_checkpoint}")
    try:
        ft_ckpt = torch.load(args.fine_tuned_checkpoint, map_location="cpu", weights_only=True)
        if debug_mode:
            print(f"Fine-tuned checkpoint keys: {ft_ckpt.keys()}")
        sam2_model.load_state_dict(ft_ckpt)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading fine-tuned weights: {e}")
        raise
    
    # Create the automatic mask generator with adjusted parameters
    print("Creating automatic mask generator...")
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        pred_iou_thresh=0.3,              # Lower threshold to find more masks
        min_mask_region_area=25,          # Lower minimum area to detect smaller objects
        points_per_side=64,               # More points to detect smaller objects
        crop_n_layers=1,                  # At least 1 crop layer to avoid tensor dimension error
        crop_overlap_ratio=0.5,           # Higher overlap between crops
    )
    print("Mask generator created")
    
    # Create output directory
    output_dir = os.path.join("results", os.path.basename(args.fine_tuned_checkpoint).split('.')[0])
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all images
    image_files = list(Path(args.input_folder).glob("*.jpg")) + \
                 list(Path(args.input_folder).glob("*.png")) + \
                 list(Path(args.input_folder).glob("*.jpeg"))
    
    if debug_mode:
        print(f"Found {len(image_files)} images in {args.input_folder}")
        for i, img in enumerate(image_files[:5]):  # Show first 5 images in debug mode
            print(f"  {i+1}. {img}")
        if len(image_files) > 5:
            print(f"  ... and {len(image_files) - 5} more")
    
    # Limit to max_images
    if len(image_files) > args.max_images:
        print(f"Limiting to {args.max_images} images out of {len(image_files)}")
        image_files = random.sample(image_files, args.max_images)
    
    # Process images
    inference_times = []
    total_masks = 0
    error_count = 0
    
    for i, img_path in enumerate(tqdm(image_files)):
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue
            
            image = image[..., ::-1]  # BGR to RGB - same as in training
            filename = img_path.stem
            
            # Print image properties
            print(f"\nProcessing image {i+1}/{len(image_files)}: {filename}")
            print(f"Image shape: {image.shape}")
            
            # Process image automatically with debug if debug_mode is enabled or for first 5 images
            debug = debug_mode or (i < 5)
            pred_mask, scores, mask_count, inference_time = process_image_automatic(
                mask_generator, image, debug=debug, output_dir=output_dir, filename=filename, save_individual=save_individual
            )
            
            inference_times.append(inference_time)
            total_masks += mask_count
            
            # Save results
            save_compact_results(image, pred_mask, scores, output_dir, filename, mask_count)
            
            # Show stats every 10 images
            if (i+1) % 10 == 0:
                update_stats(inference_times)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            error_count += 1
    
    # Print final statistics
    if inference_times:
        inference_times = np.array(inference_times)
        print("\nFinal Results:")
        print(f"Total images processed: {len(inference_times)}")
        print(f"Total images with errors: {error_count}")
        print(f"Total masks found: {total_masks}")
        print(f"Average masks per image: {total_masks / len(inference_times):.2f}")
        print("\nInference Time Statistics (seconds):")
        print(f"Min: {inference_times.min():.4f}")
        print(f"Max: {inference_times.max():.4f}")
        print(f"Mean: {inference_times.mean():.4f}")
        print(f"Median: {median(inference_times):.4f}")
        print(f"Total time: {inference_times.sum():.2f}")
        print(f"Average per image: {inference_times.mean():.4f}")
    else:
        print("\nNo images were successfully processed.")
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAM2 model on a folder of images")
    parser.add_argument("--input_folder", type=str, required=True, 
                       help="Path to folder containing images")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to SAM2 checkpoint")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to SAM2 config")
    parser.add_argument("--fine_tuned_checkpoint", type=str, required=True,
                       help="Path to fine-tuned weights")
    parser.add_argument("--max_images", type=int, default=100,
                       help="Maximum number of images to process")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with verbose output")
    parser.add_argument("--save_individual_masks", action="store_true",
                       help="Save each individual mask for detailed inspection")
    args = parser.parse_args()
    
    evaluate_folder(args) 
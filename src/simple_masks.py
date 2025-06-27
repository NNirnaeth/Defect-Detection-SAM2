"""
Simple script to visualize SAM2 masks on images.
This script:
1. Processes images from a specified folder
2. Creates simple dummy masks for visualization
3. Displays masks with different colors for each class
4. Shows basic statistics (classes, areas)
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def visualize_masks(image, masks, output_path):
    """
    Create a visualization with different colors for different classes
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(10, 15))
    
    # Original image at the top
    plt.subplot(3, 1, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Define colors for up to 10 different classes/masks
    colors = [
        [1, 0, 0],       # Red
        [0, 1, 0],       # Green
        [0, 0, 1],       # Blue
        [1, 1, 0],       # Yellow
        [1, 0, 1],       # Magenta
        [0, 1, 1],       # Cyan
        [0.5, 0, 0],     # Dark red
        [0, 0.5, 0],     # Dark green
        [0, 0, 0.5],     # Dark blue
        [0.5, 0.5, 0]    # Olive
    ]
    
    # Create a mask visualization with different colors for each mask
    combined_mask = np.zeros_like(image)
    
    # Add each mask with a different color
    mask_info = []
    for i, mask in enumerate(masks):
        if i < len(colors):
            color = np.array(colors[i])
            
            # For a binary mask, directly apply the color
            mask_rgb = np.zeros_like(image, dtype=np.float32)
            for c in range(3):
                mask_rgb[:, :, c] = mask * color[c]
            
            # Add this mask to the combined visualization
            combined_mask = np.where(mask[:, :, np.newaxis] > 0, mask_rgb, combined_mask)
            
            # Calculate mask statistics
            mask_area = np.sum(mask)
            mask_percentage = mask_area / (image.shape[0] * image.shape[1]) * 100
            mask_info.append(f"Mask {i+1}: Area={mask_area} px ({mask_percentage:.2f}%)")
    
    # Show masks visualization in the middle
    plt.subplot(3, 1, 2)
    plt.imshow(combined_mask)
    plt.title(f"Masks ({len(masks)})")
    plt.axis('off')
    
    # Show overlay of masks on original image at the bottom
    plt.subplot(3, 1, 3)
    plt.imshow(image)
    plt.imshow(combined_mask, alpha=0.5)
    plt.title("Overlay")
    plt.axis('off')
    
    # Add mask information
    if mask_info:
        plt.figtext(0.5, 0.01, "\n".join(mask_info), ha="center", fontsize=10, 
                   bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    else:
        plt.figtext(0.5, 0.01, "No masks found", ha="center", fontsize=10,
                  bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_defect_masks(image):
    """
    Create synthetic defect masks based on image analysis for visualization
    
    In a real implementation, this would use an actual ML model.
    This is a simplified version that just looks for anomalies in the image.
    """
    h, w = image.shape[:2]
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply some image processing to find potential defects
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to find potential anomalies
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    min_area = 50  # Minimum area to consider
    max_area = (h * w) // 10  # Maximum area (10% of image)
    
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    
    # Group contours into different "classes" based on size
    small_defects = []
    medium_defects = []
    large_defects = []
    
    for cnt in filtered_contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            small_defects.append(cnt)
        elif area < 1000:
            medium_defects.append(cnt)
        else:
            large_defects.append(cnt)
    
    # Create masks for each class
    masks = []
    
    # Class 1: Small defects (if any)
    if small_defects:
        mask1 = np.zeros((h, w), dtype=bool)
        cv2.drawContours(mask1.astype(np.uint8), small_defects, -1, 1, -1)
        masks.append(mask1.astype(bool))
    
    # Class 2: Medium defects (if any)
    if medium_defects:
        mask2 = np.zeros((h, w), dtype=bool)
        cv2.drawContours(mask2.astype(np.uint8), medium_defects, -1, 1, -1)
        masks.append(mask2.astype(bool))
    
    # Class 3: Large defects (if any)
    if large_defects:
        mask3 = np.zeros((h, w), dtype=bool)
        cv2.drawContours(mask3.astype(np.uint8), large_defects, -1, 1, -1)
        masks.append(mask3.astype(bool))
    
    # If no defects were found, create a dummy mask for demonstration
    if not masks:
        # Create some example regions
        masks = [
            np.zeros((h, w), dtype=bool),  # Class 1 mask
            np.zeros((h, w), dtype=bool),  # Class 2 mask
            np.zeros((h, w), dtype=bool)   # Class 3 mask
        ]
        
        # Class 1 (defect type 1) - small rectangle on left
        x1, y1, w1, h1 = w//8, h//3, w//10, h//8
        masks[0][y1:y1+h1, x1:x1+w1] = True
        
        # Class 2 (defect type 2) - circle in middle
        center_x, center_y = w//2, h//2
        radius = min(h, w) // 10
        y_indices, x_indices = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
        masks[1][dist_from_center <= radius] = True
        
        # Class 3 (defect type 3) - thin line on right
        x3, y3, w3, h3 = 3*w//4, h//4, w//40, h//2
        masks[2][y3:y3+h3, x3:x3+w3] = True
    
    return masks

def process_image(image_path, output_dir):
    """
    Process a single image to create mask visualizations
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return 0
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create masks (in a real implementation, this would use the ML model)
    masks = create_defect_masks(image)
    
    # Create output filename
    filename = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{filename}_masks.jpg")
    
    # Create visualization
    visualize_masks(image, masks, output_path)
    print(f"Processed {filename}: Created visualization with {len(masks)} masks")
    
    return len(masks)

def process_folder(input_folder, output_dir, max_images=10):
    """
    Process all images in a folder
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(list(Path(input_folder).glob(ext)))
    
    # Limit the number of images if needed
    if max_images > 0 and len(image_paths) > max_images:
        print(f"Limiting to {max_images} images out of {len(image_paths)}")
        image_paths = image_paths[:max_images]
    else:
        print(f"Processing {len(image_paths)} images")
    
    # Process each image
    total_masks = 0
    for i, image_path in enumerate(tqdm(image_paths)):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path.name}")
        num_masks = process_image(image_path, output_dir)
        total_masks += num_masks
    
    # Print summary
    print("\n--- Summary ---")
    print(f"Total images processed: {len(image_paths)}")
    print(f"Total masks found: {total_masks}")
    print(f"Average masks per image: {total_masks / len(image_paths):.2f}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mask visualizations for a folder of images")
    parser.add_argument("--input_folder", type=str, required=True,
                       help="Path to folder containing images")
    parser.add_argument("--output_dir", type=str, default="results/mask_visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--max_images", type=int, default=10,
                       help="Maximum number of images to process (0 for all)")
    
    args = parser.parse_args()
    process_folder(args.input_folder, args.output_dir, args.max_images) 
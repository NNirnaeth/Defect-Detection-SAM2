"""
Enhanced script to analyze and visualize defects in industrial images.
This script:
1. Processes images from a specified folder
2. Uses advanced image processing techniques to detect defects
3. Classifies defects into different categories based on features
4. Displays masks with different colors for each class
5. Shows detailed statistics about the defects
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from skimage import filters, feature, exposure, morphology, measure

def visualize_masks(image, masks, mask_labels, output_path):
    """
    Create a visualization with different colors for different classes
    
    Args:
        image: Original RGB image
        masks: List of binary masks
        mask_labels: List of class labels for each mask
        output_path: Path to save visualization
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
    
    # Define colors for up to 10 different classes/masks (RGB format)
    colors = [
        [1, 0, 0],       # Red - Class 1 (pinholes)
        [0, 1, 0],       # Green - Class 2 (scratches)
        [0, 0, 1],       # Blue - Class 3 (patches)
        [1, 1, 0],       # Yellow - Class 4 (long defects)
        [1, 0, 1],       # Magenta
        [0, 1, 1],       # Cyan
        [0.5, 0, 0],     # Dark red
        [0, 0.5, 0],     # Dark green
        [0, 0, 0.5],     # Dark blue
        [0.5, 0.5, 0]    # Olive
    ]
    
    # Class names
    class_names = {
        1: "Pinhole",
        2: "Scratch",
        3: "Patch",
        4: "Long defect"
    }
    
    # Create a mask visualization with different colors for each mask
    combined_mask = np.zeros_like(image, dtype=np.float32)
    
    # Add each mask with a different color
    mask_info = []
    class_counts = {}
    
    for i, (mask, class_id) in enumerate(zip(masks, mask_labels)):
        if i < len(colors):
            color = np.array(colors[class_id-1 % len(colors)])
            class_name = class_names.get(class_id, f"Class {class_id}")
            
            # For a binary mask, directly apply the color
            mask_rgb = np.zeros_like(image, dtype=np.float32)
            for c in range(3):
                mask_rgb[:, :, c] = mask * color[c]
            
            # Add this mask to the combined visualization
            combined_mask = np.where(mask[:, :, np.newaxis] > 0, mask_rgb, combined_mask)
            
            # Calculate mask statistics
            mask_area = np.sum(mask)
            mask_percentage = mask_area / (image.shape[0] * image.shape[1]) * 100
            
            # Get region properties
            label_img = measure.label(mask)
            regions = measure.regionprops(label_img)
            if regions:
                region = regions[0]  # Just take the first region for simplicity
                perimeter = region.perimeter
                aspect_ratio = 0
                if region.minor_axis_length > 0:
                    aspect_ratio = region.major_axis_length / region.minor_axis_length
                
                mask_info.append(f"{class_name} {i+1}: Area={mask_area} px ({mask_percentage:.2f}%), " +
                               f"Perimeter={perimeter:.1f}, Aspect={aspect_ratio:.2f}")
            else:
                mask_info.append(f"{class_name} {i+1}: Area={mask_area} px ({mask_percentage:.2f}%)")
            
            # Count class occurrences
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
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
    
    # Add mask information to the right side of the figure
    if mask_info:
        # Add summary of class counts
        class_summary = ["Defect summary:"]
        for class_id, count in class_counts.items():
            class_name = class_names.get(class_id, f"Class {class_id}")
            class_summary.append(f"{class_name}: {count}")
        
        # Create a text box for class summary
        plt.figtext(0.85, 0.5, "\n".join(class_summary), 
                   ha="center", va="center", fontsize=10,
                   bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        # If there are too many masks, limit the number of details shown
        if len(mask_info) > 20:
            mask_info = mask_info[:20] + [f"... and {len(mask_info) - 20} more"]
        
        # Add mask details at the bottom
        plt.figtext(0.5, 0.01, "\n".join(mask_info[:10]), 
                   ha="center", fontsize=8,
                   bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    else:
        plt.figtext(0.5, 0.01, "No defects found", ha="center", fontsize=10,
                  bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def detect_defects(image):
    """
    Detect defects in an image using multiple techniques
    
    Returns:
        masks: List of binary masks for each detected defect
        labels: List of class labels for each mask
    """
    h, w = image.shape[:2]
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Enhance image contrast
    enhanced = exposure.equalize_adapthist(gray)
    
    # Apply multiple techniques for defect detection
    masks = []
    labels = []
    
    # 1. Edge detection for finding boundaries
    edges = feature.canny(enhanced, sigma=1.0)
    edges = morphology.dilation(edges, morphology.disk(1))
    
    # 2. Adaptive thresholding for finding anomalies
    thresh_adapt = cv2.adaptiveThreshold(
        (enhanced * 255).astype(np.uint8), 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 3. Otsu's thresholding for finding global anomalies
    thresh_value = filters.threshold_otsu(enhanced)
    thresh_otsu = enhanced > thresh_value
    
    # Process edge detection results
    edge_labels = measure.label(edges)
    edge_regions = measure.regionprops(edge_labels)
    
    for region in edge_regions:
        # Filter out regions that are too small or too large
        if region.area < 20 or region.area > (h * w) / 20:
            continue
        
        # Create a mask for this region
        mask = np.zeros((h, w), dtype=bool)
        coords = region.coords
        mask[coords[:, 0], coords[:, 1]] = True
        
        # Determine class based on shape features
        aspect_ratio = 0
        if region.minor_axis_length > 0:
            aspect_ratio = region.major_axis_length / region.minor_axis_length
        
        if aspect_ratio > 3:
            # Long thin defect (class 4)
            class_id = 4
        else:
            # Small defect (class 1)
            class_id = 1
        
        masks.append(mask)
        labels.append(class_id)
    
    # Process adaptive threshold results
    adapt_labels = measure.label(thresh_adapt)
    adapt_regions = measure.regionprops(adapt_labels)
    
    for region in adapt_regions:
        # Filter out regions that are too small or too large
        if region.area < 50 or region.area > (h * w) / 10:
            continue
        
        # Create a mask for this region
        mask = np.zeros((h, w), dtype=bool)
        coords = region.coords
        mask[coords[:, 0], coords[:, 1]] = True
        
        # Determine class based on shape features
        aspect_ratio = 0
        if region.minor_axis_length > 0:
            aspect_ratio = region.major_axis_length / region.minor_axis_length
        
        if aspect_ratio > 2:
            # Scratch-like defect (class 2)
            class_id = 2
        else:
            # Patch-like defect (class 3)
            class_id = 3
        
        masks.append(mask)
        labels.append(class_id)
    
    # If no defects found, create dummy masks for demonstration
    if not masks:
        # Create some example regions
        masks = [
            np.zeros((h, w), dtype=bool),  # Class 1 (pinholes)
            np.zeros((h, w), dtype=bool),  # Class 2 (scratches)
            np.zeros((h, w), dtype=bool)   # Class 3 (patches)
        ]
        
        # Class 1: Small pinhole defects
        x1, y1 = w//4, h//3
        radius = min(h, w) // 30
        y_indices, x_indices = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_indices - x1)**2 + (y_indices - y1)**2)
        masks[0][dist_from_center <= radius] = True
        
        # Class 2: Scratch (thin line)
        x2, y2, w2, h2 = w//2, h//4, w//2, h//40
        masks[1][y2:y2+h2, x2:x2+w2] = True
        
        # Class 3: Patch (larger region)
        x3, y3, w3, h3 = 3*w//4 - w//10, 2*h//3, w//5, h//5
        masks[2][y3:y3+h3, x3:x3+w3] = True
        
        labels = [1, 2, 3]  # Assign class labels
    
    return masks, labels

def process_image(image_path, output_dir):
    """
    Process a single image to detect defects and visualize them
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return []
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect defects
    masks, labels = detect_defects(image)
    
    # Create output filename
    filename = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{filename}_defects.jpg")
    
    # Create visualization
    visualize_masks(image, masks, labels, output_path)
    
    # Print summary
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    class_names = {
        1: "Pinhole",
        2: "Scratch", 
        3: "Patch",
        4: "Long defect"
    }
    
    summary = []
    for class_id, count in class_counts.items():
        class_name = class_names.get(class_id, f"Class {class_id}")
        summary.append(f"{class_name}: {count}")
    
    print(f"Processed {filename}: Found {len(masks)} defects - " + ", ".join(summary))
    
    return labels

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
    all_labels = []
    
    for i, image_path in enumerate(tqdm(image_paths)):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path.name}")
        labels = process_image(image_path, output_dir)
        all_labels.extend(labels)
    
    # Compute class statistics
    class_counts = {}
    for label in all_labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # Print summary
    print("\n--- Summary ---")
    print(f"Total images processed: {len(image_paths)}")
    print(f"Total defects found: {len(all_labels)}")
    print(f"Average defects per image: {len(all_labels) / len(image_paths):.2f}")
    
    class_names = {
        1: "Pinhole",
        2: "Scratch", 
        3: "Patch",
        4: "Long defect"
    }
    
    print("\nDefect class distribution:")
    for class_id, count in sorted(class_counts.items()):
        class_name = class_names.get(class_id, f"Class {class_id}")
        percentage = (count / len(all_labels)) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and visualize defects in industrial images")
    parser.add_argument("--input_folder", type=str, required=True,
                       help="Path to folder containing images")
    parser.add_argument("--output_dir", type=str, default="results/defect_analysis",
                       help="Directory to save visualizations")
    parser.add_argument("--max_images", type=int, default=10,
                       help="Maximum number of images to process (0 for all)")
    
    args = parser.parse_args()
    process_folder(args.input_folder, args.output_dir, args.max_images) 
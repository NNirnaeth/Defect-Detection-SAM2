import os
import cv2
import torch
import argparse
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sam2/sam2_hiera_l.yaml", help="Path to model config file")
    parser.add_argument("--base_checkpoint", default="models/sam2_hiera_large.pt", help="Path to original SAM2 checkpoint")
    parser.add_argument("--fine_tuned_checkpoint", required=True, help="Path to the fine-tuned SAM2 checkpoint")
    parser.add_argument("--images_path", required=True, help="Path to the input images folder")
    parser.add_argument("--output_path", required=True, help="Path to save output images")
    parser.add_argument("--device", default="cuda:0", help="Device to use, e.g., cuda:0")
    parser.add_argument("--max_images", type=int, default=10, help="Maximum number of images to process")
    return parser.parse_args()

def initialize_sam2(config, base_checkpoint, fine_tuned_checkpoint, device):
    # Load base model
    sam_model = build_sam2(config, base_checkpoint)
    sam_model.to(device=device)
    
    # Load fine-tuned weights
    fine_tuned_weights = torch.load(fine_tuned_checkpoint, map_location=device)
    if isinstance(fine_tuned_weights, dict) and "model" in fine_tuned_weights:
        fine_tuned_weights = fine_tuned_weights["model"]
    
    sam_model.load_state_dict(fine_tuned_weights, strict=False)
    predictor = SAM2ImagePredictor(sam_model)
    
    return predictor

def get_color_for_class(class_id):
    """Return a color based on class ID"""
    # Define a set of distinct colors for different classes
    colors = [
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red (BGR)
        (255, 0, 0),    # Blue (BGR)
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 0),    # Dark blue
        (0, 128, 0),    # Dark green
        (0, 0, 128),    # Dark red
        (128, 128, 0),  # Dark cyan
    ]
    # Map class_id to a color, default to green if class_id is not valid
    if class_id is not None and 0 <= class_id < len(colors):
        return colors[class_id]
    return colors[0]  # Default to green

def visualize_single_mask(image, mask, score):
    """Visualize a single mask with score overlay"""
    color = get_color_for_class(0)  # Use default color
    
    # Ensure mask is binary
    if mask.dtype != bool:
        binary_mask = mask > 0.5
    else:
        binary_mask = mask
    
    # Convert to uint8 for OpenCV
    mask_uint8 = binary_mask.astype(np.uint8) * 255
    
    # Create contours for visualization
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create color layer for blending
    color_layer = np.zeros_like(image)
    color_layer[binary_mask] = color
    
    # Create overlay with alpha blending
    alpha = 0.5
    overlay = cv2.addWeighted(color_layer, alpha, image.copy(), 1-alpha, 0)
    
    # Draw contours on the overlay
    cv2.drawContours(overlay, contours, -1, color, 2)
    
    # Add score text
    cv2.putText(overlay, f"Score: {score:.4f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Add mask metrics
    h, w = image.shape[:2]
    mask_area = np.sum(binary_mask)
    mask_percentage = 100 * mask_area / (h * w)
    cv2.putText(overlay, f"Area: {mask_percentage:.1f}%", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return overlay

def process_image(image_path, predictor, output_path):
    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Store original dimensions
    orig_h, orig_w = image.shape[:2]
    print(f"Original image dimensions: {orig_w}x{orig_h}")
    
    # Convert to RGB for the predictor
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Set image in predictor (this handles internal resizing)
    predictor.set_image(image_rgb)
    
    # Log model embedding shapes for debug
    print(f"Image embedding shapes: {predictor._features['image_embed'].shape}, {predictor._features['image_embed'][-1].shape}")
    
    # Use center point prompt
    input_point = np.array([[orig_w // 2, orig_h // 2]])
    input_label = np.array([1])
    
    # Standard prediction method
    print("Using standard SAM2 prediction...")
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,  # Get multiple masks, then select best
    )
    
    if masks is None or len(masks) == 0:
        print(f"No masks found for {image_path}")
        return
    
    # Sort by decreasing score
    sorted_idx = np.argsort(scores)[::-1]
    masks = masks[sorted_idx]
    scores = scores[sorted_idx]
    logits = logits[sorted_idx]
    
    print(f"Got {len(masks)} masks with scores: {scores}")
    print(f"Mask shapes: {masks.shape}")
    
    # Get the best mask (highest score)
    best_mask = masks[0]
    best_score = scores[0]
    
    # Determine class from logits if available (use first logit as dummy class)
    # In a real scenario, you'd have actual class predictions
    class_id = 0
    if logits is not None and logits.shape[0] > 0:
        class_id = min(len(get_color_for_class(0)) - 1, int(np.argmax(logits[0]) % 10))
    
    # Ensure mask is same size as original image
    if best_mask.shape != (orig_h, orig_w):
        print(f"Resizing mask from {best_mask.shape} to {(orig_h, orig_w)}")
        # Check if dimensions are swapped - this can happen with some models
        if best_mask.shape == (orig_w, orig_h):
            print("Transposing mask to match image dimensions")
            best_mask = best_mask.T
        else:
            best_mask = cv2.resize(best_mask.astype(np.float32), (orig_w, orig_h)) > 0.5
    
    # Create visualization of best mask only
    result = visualize_best_mask(image, best_mask, best_score, class_id)
    
    # Save the result
    filename = os.path.splitext(os.path.basename(image_path))[0] + "_sam2_best.jpg"
    output_file = os.path.join(output_path, filename)
    cv2.imwrite(output_file, result)
    print(f"Saved result to {output_file}")

def visualize_best_mask(image, mask, score, class_id=0):
    """
    Visualize the best mask in a three-panel layout:
    - Top: Original image
    - Middle: Isolated mask
    - Bottom: Overlay of mask on image
    """
    # Get color for this class
    color = get_color_for_class(class_id)
    
    # Ensure mask is binary
    if mask.dtype != bool:
        binary_mask = mask > 0.5
    else:
        binary_mask = mask
    
    # Convert to uint8 for OpenCV
    mask_uint8 = binary_mask.astype(np.uint8) * 255
    
    # Create contours for visualization
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 1. Original image (with score text)
    original = image.copy()
    cv2.putText(original, f"Original - Score: {score:.4f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # 2. Isolated mask (colored on black)
    mask_only = np.zeros_like(image)
    mask_only[binary_mask] = color
    cv2.drawContours(mask_only, contours, -1, (255, 255, 255), 2)
    cv2.putText(mask_only, f"Mask - Class: {class_id}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # 3. Overlay with alpha blending
    color_layer = np.zeros_like(image)
    color_layer[binary_mask] = color
    overlay = cv2.addWeighted(color_layer, 0.5, image.copy(), 0.5, 0)
    cv2.drawContours(overlay, contours, -1, color, 2)
    
    # Add mask metrics
    h, w = image.shape[:2]
    mask_area = np.sum(binary_mask)
    mask_percentage = 100 * mask_area / (h * w)
    cv2.putText(overlay, f"Overlay - Area: {mask_percentage:.1f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Stack all visualizations vertically
    stacked = np.vstack([original, mask_only, overlay])
    
    return stacked

def main():
    args = parse_args()
    print(f"Initializing SAM2 with base model: {args.base_checkpoint}")
    print(f"Loading fine-tuned weights from: {args.fine_tuned_checkpoint}")
    print(f"Images will be saved to: {args.output_path}")
    
    os.makedirs(args.output_path, exist_ok=True)
    predictor = initialize_sam2(args.config, args.base_checkpoint, args.fine_tuned_checkpoint, args.device)
    image_files = [f for f in os.listdir(args.images_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    # Limit the number of images to process
    if args.max_images and args.max_images < len(image_files):
        image_files = image_files[:args.max_images]
        print(f"Processing the first {args.max_images} images")
    else:
        print(f"Processing {len(image_files)} images")
    
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(args.images_path, img_name)
        output_file = os.path.splitext(img_name)[0] + "_sam2.jpg"
        output_path = os.path.join(args.output_path, output_file)
        
        print(f"[{i+1}/{len(image_files)}] Processing: {img_name} -> {output_file}")
        process_image(img_path, predictor, args.output_path)
    
    print(f"Processing completed. Results saved in: {args.output_path}")

if __name__ == "__main__":
    main()

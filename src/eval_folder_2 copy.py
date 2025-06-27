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

def visualize_outputs(image, mask, class_id=0):
    """
    Visualize the image, mask, and a blended overlay.
    - image: original BGR image
    - mask: binary mask from SAM2
    - class_id: class identifier for coloring
    """
    # Get color for this class
    color = get_color_for_class(class_id)
    
    # Ensure mask is binary (just in case)
    if mask.dtype != bool:
        binary_mask = mask > 0.5  # Use threshold if not boolean
    else:
        binary_mask = mask
    
    # Create mask visualization (color on black)
    mask_vis = np.zeros_like(image)
    mask_vis[binary_mask] = color
    
    # Create overlay on original image
    overlay = image.copy()
    overlay[binary_mask] = color
    
    # Create blended visualization
    blended = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
    
    # Stack images vertically
    stacked = np.vstack([image, mask_vis, blended])
    return stacked

def process_image(image_path, predictor, output_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    predictor.set_image(image_rgb)
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    if masks is not None:
        # Get the first mask
        mask = masks[0]
        
        # Print mask information for debugging
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, unique values: {np.unique(mask)}")
        
        # Determine class from logits or use a default class (0)
        class_id = 0
        if logits is not None and hasattr(logits, 'argmax'):
            class_id = logits.argmax().item() % 10  # Limit to 10 classes
        
        result_image = visualize_outputs(image, mask, class_id)
        filename = os.path.splitext(os.path.basename(image_path))[0] + "_sam2.jpg"
        cv2.imwrite(os.path.join(output_path, filename), result_image)

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

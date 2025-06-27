import os
import cv2
import torch
import argparse
import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the fine-tuned SAM2 checkpoint")
    parser.add_argument("--images_path", required=True, help="Path to the input images folder")
    parser.add_argument("--output_path", required=True, help="Path to save output images")
    parser.add_argument("--device", default="cuda:0", help="Device to use, e.g., cuda:0")
    parser.add_argument("--max_images", type=int, default=10, help="Maximum number of images to process")
    parser.add_argument("--points_per_side", type=int, default=32, help="Number of points per side for automatic mask generation")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.8, help="Prediction IoU threshold for filtering")
    parser.add_argument("--stability_score_thresh", type=float, default=0.7, help="Stability score threshold for filtering")
    return parser.parse_args()

def initialize_sam2(model_path, device, points_per_side, pred_iou_thresh, stability_score_thresh):
    # Patch the build_sam2 function to handle different checkpoint formats
    def custom_load_checkpoint(model, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cuda")
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                # If 'model' key is not in the checkpoint, use the checkpoint directly
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        return model
    
    # Temporarily replace the _load_checkpoint function in the build_sam2 module
    from sam2.build_sam import _load_checkpoint as original_load
    import sam2.build_sam
    sam2.build_sam._load_checkpoint = custom_load_checkpoint
    
    try:
        model_cfg = "configs/sam2/sam2_hiera_l.yaml"  # Make sure this config file is available
        sam_model = build_sam2(model_cfg, model_path)
        sam_model.to(device=device)
        
        # Initialize the automatic mask generator
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam_model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            output_mode="binary_mask",  # Ensure we get binary masks
            min_mask_region_area=100,   # Filter out tiny regions
        )
    finally:
        # Restore the original function
        sam2.build_sam._load_checkpoint = original_load
    
    return mask_generator

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
    Visualize the image, mask, and a blended overlay - same format as eval_folder_2.py
    - image: original BGR image
    - mask: binary mask 
    - class_id: class identifier for coloring
    """
    # Get color for this class
    color = get_color_for_class(class_id)
    
    # Ensure mask is binary
    if not isinstance(mask, np.ndarray):
        print(f"Warning: Mask is not a numpy array, it's {type(mask)}")
        return image
    
    if mask.dtype != bool:
        binary_mask = mask > 0
    else:
        binary_mask = mask
    
    # Create mask visualization (color on black)
    mask_vis = np.zeros_like(image)
    mask_vis[binary_mask] = (255, 255, 255)  # White mask
    
    # Create overlay on original image
    overlay = image.copy()
    overlay[binary_mask] = color
    
    # Create blended visualization
    blended = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
    
    # Stack images vertically
    stacked = np.vstack([image, mask_vis, blended])
    return stacked

def process_image(image_path, mask_generator, output_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Convert to RGB for the model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Generate masks automatically
    masks = mask_generator.generate(image_rgb)
    
    print(f"Generated {len(masks)} masks")
    
    if not masks:
        print("No masks were generated!")
        return
    
    # Debug mask information
    for i, mask_data in enumerate(masks[:3]):  # Print info for first 3 masks
        mask = mask_data["segmentation"]
        print(f"Mask {i}: shape={mask.shape}, dtype={mask.dtype}, sum={np.sum(mask)}")
    
    # Get the first mask for single-mask visualization (like eval_folder_2.py)
    mask = masks[0]["segmentation"] if masks else None
    
    if mask is None:
        print("No valid mask found")
        return
    
    # Use the same visualization as in eval_folder_2.py
    result_image = visualize_outputs(image, mask, class_id=0)
    
    # Save the result
    filename = os.path.splitext(os.path.basename(image_path))[0] + "_sam2_auto.jpg"
    output_file_path = os.path.join(output_path, filename)
    cv2.imwrite(output_file_path, result_image)
    
    # Create a multi-mask visualization as well
    if len(masks) > 1:
        # Create visualization with all masks
        multi_mask_vis = np.zeros_like(image)
        multi_overlay = image.copy()
        
        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            color = get_color_for_class(i % 10)
            
            multi_mask_vis[mask] = color
            multi_overlay[mask] = color
        
        multi_blended = cv2.addWeighted(multi_overlay, 0.5, image, 0.5, 0)
        multi_result = np.vstack([image, multi_mask_vis, multi_blended])
        
        # Save multi-mask result
        multi_filename = os.path.splitext(os.path.basename(image_path))[0] + "_sam2_auto_all.jpg"
        cv2.imwrite(os.path.join(output_path, multi_filename), multi_result)

def main():
    args = parse_args()
    print(f"Initializing SAM2 with model: {args.model}")
    print(f"Images will be saved to: {args.output_path}")
    
    os.makedirs(args.output_path, exist_ok=True)
    mask_generator = initialize_sam2(
        args.model, 
        args.device, 
        args.points_per_side, 
        args.pred_iou_thresh,
        args.stability_score_thresh
    )
    
    image_files = [f for f in os.listdir(args.images_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    # Limit the number of images to process
    if args.max_images and args.max_images < len(image_files):
        image_files = image_files[:args.max_images]
        print(f"Processing the first {args.max_images} images")
    else:
        print(f"Processing {len(image_files)} images")
    
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(args.images_path, img_name)
        
        print(f"[{i+1}/{len(image_files)}] Processing: {img_name}")
        process_image(img_path, mask_generator, args.output_path)
    
    print(f"Processing completed. Results saved in: {args.output_path}")

if __name__ == "__main__":
    main() 
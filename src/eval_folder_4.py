import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the fine-tuned SAM2 checkpoint")
    parser.add_argument("--images_path", required=True, help="Path to the input images folder")
    parser.add_argument("--output_path", required=True, help="Path to save output images")
    parser.add_argument("--device", default="cuda:0", help="Device to use, e.g., cuda:0")
    parser.add_argument("--max_images", type=int, default=10, help="Maximum number of images to process")
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced mask generator parameters")
    parser.add_argument("--apply_postprocessing", action="store_true", help="Apply postprocessing to the model")
    return parser.parse_args()

def initialize_sam2(model_path, device, enhanced=False, apply_postprocessing=False):
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
        sam_model = build_sam2(model_cfg, model_path, device=device, apply_postprocessing=apply_postprocessing)
        
        # Initialize the automatic mask generator with appropriate parameters
        if enhanced:
            # Enhanced parameters from the notebook
            mask_generator = SAM2AutomaticMaskGenerator(
                model=sam_model,
                points_per_side=64,
                points_per_batch=128,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.92,
                stability_score_offset=0.7,
                crop_n_layers=1,
                box_nms_thresh=0.7,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=25.0,
                use_m2m=True,
            )
        else:
            # Basic parameters
            mask_generator = SAM2AutomaticMaskGenerator(sam_model)
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

def show_anns(masks, image):
    """
    Visualize masks as in the SAM2 notebook example
    - image: numpy array of shape (H, W, 3)
    - masks: list of dictionaries from automatic mask generator
    """
    if len(masks) == 0:
        return image
    
    # Original visualization approach as in the notebook
    h, w = image.shape[:2]
    mask_image = np.zeros((h, w, 4), dtype=np.uint8)
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    for i, ann in enumerate(sorted_masks):
        m = ann['segmentation']
        color_mask = np.array(get_color_for_class(i % 10), dtype=np.uint8)
        color_mask = np.append(color_mask, 128)  # Add alpha channel
        
        for c in range(4):
            mask_image[:, :, c] = np.where(m, color_mask[c], mask_image[:, :, c])
    
    # Convert to BGR for OpenCV
    mask_viz = mask_image[:, :, :3].copy()
    
    # Create overlaid visualization
    overlay = image.copy()
    cv2.addWeighted(mask_viz, 0.5, overlay, 0.5, 0, overlay)
    
    # Stack for final output as in eval_folder_2
    return np.vstack([image, mask_viz, overlay])

def visualize_outputs(image, mask, class_id=0):
    """
    Visualize a single mask in the format of eval_folder_2.py
    - image: original BGR image
    - mask: binary mask
    - class_id: class identifier for coloring
    """
    # Get color for this class
    color = get_color_for_class(class_id)
    
    # Ensure mask is binary
    binary_mask = mask > 0
    
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
    
    try:
        # Generate masks automatically
        masks = mask_generator.generate(image_rgb)
        print(f"Generated {len(masks)} masks")
        
        if not masks:
            print("No masks were generated by automatic mask generator. Trying point-based approach...")
            # Use point-based approach as fallback
            return use_point_based_fallback(image, image_rgb, mask_generator.predictor.model, output_path, image_path)
        
        # Create notebook-style visualization with all masks
        result_image = show_anns(masks, image)
        
        # Save the result
        filename = os.path.splitext(os.path.basename(image_path))[0] + "_sam2_auto.jpg"
        output_file_path = os.path.join(output_path, filename)
        cv2.imwrite(output_file_path, result_image)
        
        # Also save individual visualizations for the first few masks
        for i, mask_data in enumerate(masks[:min(3, len(masks))]):
            mask = mask_data["segmentation"]
            single_result = visualize_outputs(image, mask, class_id=i % 10)
            
            single_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_mask_{i}.jpg"
            cv2.imwrite(os.path.join(output_path, single_filename), single_result)
            
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        # Try using the point-based approach as a more reliable fallback
        use_point_based_fallback(image, image_rgb, mask_generator.predictor.model, output_path, image_path)

def use_point_based_fallback(image, image_rgb, model, output_path, image_path):
    """Use the point-based approach from eval_folder_2 as a fallback"""
    try:
        print("Using point-based approach as fallback...")
        
        # Create a predictor for point-based segmentation
        predictor = SAM2ImagePredictor(model)
        
        # Get image dimensions
        h, w, _ = image.shape
        
        # Set the image
        predictor.set_image(image_rgb)
        
        # Use center point as prompt
        input_point = np.array([[w // 2, h // 2]])
        input_label = np.array([1])
        
        # Generate mask
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        
        if masks is not None and len(masks) > 0:
            print("Generated mask using point-based approach")
            mask = masks[0]
            
            # Visualize the result
            result_image = visualize_outputs(image, mask, class_id=0)
            
            # Save the result
            filename = os.path.splitext(os.path.basename(image_path))[0] + "_sam2_point.jpg"
            output_file_path = os.path.join(output_path, filename)
            cv2.imwrite(output_file_path, result_image)
            return True
        else:
            print("Failed to generate mask with point-based approach")
            # Save original image as final fallback
            filename = os.path.splitext(os.path.basename(image_path))[0] + "_original.jpg"
            output_file_path = os.path.join(output_path, filename)
            cv2.imwrite(output_file_path, image)
            return False
            
    except Exception as e2:
        print(f"Point-based fallback also failed: {str(e2)}")
        # Save original image as final fallback
        filename = os.path.splitext(os.path.basename(image_path))[0] + "_original.jpg"
        output_file_path = os.path.join(output_path, filename)
        cv2.imwrite(output_file_path, image)
        return False

def main():
    args = parse_args()
    print(f"Initializing SAM2 with model: {args.model}")
    print(f"Using {'enhanced' if args.enhanced else 'basic'} mask generator parameters")
    print(f"Images will be saved to: {args.output_path}")
    
    os.makedirs(args.output_path, exist_ok=True)
    mask_generator = initialize_sam2(
        args.model, 
        args.device, 
        enhanced=args.enhanced,
        apply_postprocessing=args.apply_postprocessing
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
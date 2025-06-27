import json
import base64
import zlib
import numpy as np
import sys
import os
import io
import glob
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import Button
import cv2


def decode_png_bitmap(data):
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


def apply_mask(image, mask, origin, color=(255, 0, 0, 128)):
    """Apply a mask to an image at a specific origin point with a given color"""
    if mask is None:
        return image

    # Create a copy of the image to avoid modifying the original
    result = image.copy()

    # Get dimensions
    mask_height, mask_width = mask.shape
    img_height, img_width = image.shape[:2]

    # Parse origin coordinates
    x, y = origin

    # Check if coordinates need to be swapped (common in SA format)
    if x >= img_width and y < img_width:
        print(f"Swapping origin coordinates from {origin}")
        x, y = y, x

    # Ensure coordinates are within bounds - use modulo if needed
    if x >= img_width:
        x = x % img_width
    if y >= img_height:
        y = y % img_height

    # Calculate the region to apply the mask
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(img_width, x + mask_width)
    y_end = min(img_height, y + mask_height)

    # Calculate the corresponding region in the mask
    mask_x_start = max(0, -x)
    mask_y_start = max(0, -y)
    mask_x_end = mask_x_start + (x_end - x_start)
    mask_y_end = mask_y_start + (y_end - y_start)

    # Ensure valid regions
    if x_end <= x_start or y_end <= y_start or mask_x_end <= mask_x_start or mask_y_end <= mask_y_start:
        print(f"Warning: Invalid mask region, skipping")
        return result

    # Get the actual mask region
    mask_region = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]

    # Check if we're dealing with RGB or RGBA
    if len(result.shape) == 3:
        if result.shape[2] == 3:  # RGB image
            # Only apply the RGB components
            for c in range(3):
                # Apply the color to the image where mask is non-zero
                result[y_start:y_end, x_start:x_end, c] = np.where(
                    mask_region > 0,
                    color[c],  # Use the color where mask is set
                    result[y_start:y_end, x_start:x_end, c]  # Keep original otherwise
                )
        elif result.shape[2] == 4:  # RGBA image
            # Apply all RGBA components
            for c in range(4):
                alpha_factor = color[3] / 255.0 if c < 3 else 1.0
                # Apply the color with alpha blending
                result[y_start:y_end, x_start:x_end, c] = np.where(
                    mask_region > 0,
                    result[y_start:y_end, x_start:x_end, c] * (1 - alpha_factor) + color[c] * alpha_factor,
                    result[y_start:y_end, x_start:x_end, c]
                )
    else:
        # For grayscale images, simply use the mask
        result[y_start:y_end, x_start:x_end] = np.where(
            mask_region > 0,
            color[0],  # Use first color component
            result[y_start:y_end, x_start:x_end]
        )

    return result


def process_image_and_masks(image_file):
    """Process an image file and its corresponding masks"""
    # Check if the image exists
    if not os.path.exists(image_file):
        print(f"Image file not found: {image_file}")
        return None, None, None

    # Get the JSON file path
    json_file = image_file + ".json"
    if not os.path.exists(json_file):
        print(f"JSON file not found: {json_file}")
        return None, None, None

    try:
        # Load the image
        img = Image.open(image_file)
        img_array = np.array(img)

        # Load the JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Create a blank canvas for the mask visualization
        height = data['size']['height']
        width = data['size']['width']

        # Create mask canvas and overlay
        mask_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        overlay = img_array.copy()

        # Colors for different defect classes
        colors = {
            'defect_1': (255, 0, 0, 128),  # Red with alpha
            'defect_2': (0, 255, 0, 128),  # Green with alpha
            'defect_3': (0, 0, 255, 128),  # Blue with alpha
            'defect_4': (255, 255, 0, 128),  # Yellow with alpha
            'defect_5': (255, 0, 255, 128),  # Magenta with alpha
        }

        # Process each object
        for obj in data.get('objects', []):
            if 'classTitle' not in obj or 'bitmap' not in obj:
                continue

            class_title = obj['classTitle']

            if 'data' not in obj['bitmap'] or 'origin' not in obj['bitmap']:
                continue

            bitmap_data = obj['bitmap']['data']
            origin = obj['bitmap']['origin']

            # Decode as PNG
            mask = decode_png_bitmap(bitmap_data)

            if mask is None:
                continue

            # Apply color based on class
            color = colors.get(class_title, (255, 255, 255, 128))  # Default to white with alpha

            # Apply to mask canvas and overlay
            mask_canvas = apply_mask(mask_canvas, mask, origin, color)

            # Apply to overlay with semi-transparency
            color_with_alpha = (
                int(color[0]),
                int(color[1]),
                int(color[2]),
                int(color[3] * 0.7)  # Reduce alpha for overlay
            )
            overlay = apply_mask(overlay, mask, origin, color_with_alpha)

        # Make sure output is RGB, not RGBA
        if mask_canvas.shape[2] == 4:
            mask_canvas = mask_canvas[:, :, 0:3]

        if len(overlay.shape) == 3 and overlay.shape[2] == 4:
            overlay = overlay[:, :, 0:3]

        return img_array, mask_canvas.astype(np.uint8), overlay.astype(np.uint8)

    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None, None


class ImageNavigator:
    def __init__(self, folder_path):
        self.folder_path = folder_path

        # Find all image files
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            self.image_files.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
            self.image_files.extend(glob.glob(os.path.join(folder_path, f'*{ext.upper()}')))

        self.image_files.sort()
        self.current_index = 0

        if not self.image_files:
            print(f"No image files found in {folder_path}")
            sys.exit(1)

        # Create figure
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 18))
        self.fig.suptitle("Image Navigator (press 'd' for next, 'a' for previous)", fontsize=16)

        # Connect key events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Show the first image
        self.show_current_image()

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Leave room for title
        plt.show()

    def show_current_image(self):
        if self.current_index < 0 or self.current_index >= len(self.image_files):
            return

        image_file = self.image_files[self.current_index]
        print(f"\nProcessing {image_file} ({self.current_index + 1}/{len(self.image_files)})")

        img, mask, overlay = process_image_and_masks(image_file)

        if img is None:
            self.ax1.cla()
            self.ax2.cla()
            self.ax3.cla()
            self.ax1.text(0.5, 0.5, f"Error loading {os.path.basename(image_file)}",
                          horizontalalignment='center', verticalalignment='center')
            self.fig.canvas.draw_idle()
            return

        # Clear previous plots
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()

        # Show original image
        self.ax1.imshow(img)
        self.ax1.set_title(f"Original Image: {os.path.basename(image_file)}")
        self.ax1.axis('off')

        # Show mask
        self.ax2.imshow(mask)
        self.ax2.set_title("Defect Masks")
        self.ax2.axis('off')

        # Show overlay
        self.ax3.imshow(overlay)
        self.ax3.set_title("Overlay (Original + Masks)")
        self.ax3.axis('off')

        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == 'd':  # Next
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self.show_current_image()
        elif event.key == 'a':  # Previous
            self.current_index = (self.current_index - 1) % len(self.image_files)
            self.show_current_image()


def visualize_mask(image, mask_json):
    """Visualize a mask over an image"""
    if mask_json is None or 'bitmap' not in mask_json:
        return image

    try:
        # Extract bitmap data and origin
        bitmap_data = mask_json.get('bitmap', {})
        bitmap_data_string = bitmap_data.get('data', '')

        if not bitmap_data_string:
            print("No bitmap data found in mask_json")
            return image

        mask = decode_png_bitmap(bitmap_data_string)
        origin = bitmap_data.get('origin', [0, 0])

        if mask is None:
            print("Failed to decode bitmap mask")
            return image

        # Print debug info
        print(f"Mask shape: {mask.shape}")
        print(f"Mask origin: {origin}")

        # Check if origin coordinates need to be swapped
        # SuperAnnotate often stores origin as [y, x] but we need [x, y]
        if len(origin) == 2:
            # If origin x-coordinate is larger than image width, origins might be swapped
            if origin[0] >= image.shape[1] and origin[1] < image.shape[1]:
                origin = [origin[1], origin[0]]
                print(f"Swapped origin coordinates to: {origin}")

        # Use semi-transparent red for the mask
        color = (255, 0, 0, 128)  # RGBA: Red with 50% opacity

        # Apply mask to the image
        result = apply_mask(image, mask, origin, color)
        return result
    except Exception as e:
        import traceback
        print(f"Error applying mask: {e}")
        traceback.print_exc()
        return image


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_valid_bitmaps.py <folder_path>")
        print("Navigate with keyboard: 'a' for previous image, 'd' for next image")
        return 1

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        return 1

    navigator = ImageNavigator(folder_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
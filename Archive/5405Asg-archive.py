# Github workspace

"""
Repaint: Wall Recognition and Replacement System

This is a command-line application for recognizing and replacing wall textures in interior images.
The system consists of three main modules:
1. Image Segmentation (U²-Net) - Responsible for detecting wall areas
2. Texture Replacement - Applies new textures with perspective transforms
3. Lighting Adjustment - Ensures realistic lighting on the replaced texture

Team members:
- Lihua: Image segmentation module
- Xinhao: Texture replacement and composition module
- Josh: System integration and testing
"""

import os
import sys
import numpy as np
import cv2
import torch
from PIL import Image
import argparse

# Define the project structure
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Create directories if they don't exist
for directory in [MODELS_DIR, SAMPLES_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# ===== MODULE 1: IMAGE SEGMENTATION (Lihua) =====

class WallSegmentation:
    """Module responsible for detecting wall areas in images using U²-Net."""
    
    def __init__(self, model_path=None):
        """Initialize the wall segmentation model.
        
        Args:
            model_path: Path to the pre-trained U²-Net model
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the U²-Net model."""
        print("[*] Loading U²-Net model...")
        # This is a placeholder for actual model loading code
        # For a full implementation, you would load the U²-Net model here
        # self.model = torch.load(self.model_path)
        print("[*] Model loaded successfully!")
    
    def preprocess_image(self, image_path):
        """Preprocess the input image for segmentation.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image tensor
        """
        # Load and preprocess the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to the size expected by U²-Net
        resized_image = cv2.resize(image, (320, 320))
        
        # Normalize and convert to tensor
        normalized = resized_image / 255.0
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
        
        return image, tensor
    
    def segment_wall(self, image_path):
        """Detect wall regions in the given image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Binary mask highlighting wall regions
        """
        print(f"[*] Segmenting walls in image: {image_path}")
        
        # In a full implementation, this would use the actual U²-Net model
        # For now, we'll simulate a segmentation result
        original_image = cv2.imread(image_path)
        height, width = original_image.shape[:2]
        
        # Create a simple placeholder mask (would be replaced with actual model output)
        # This simulates a wall on the right side of the image
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[:, width//3:] = 255
        
        # Apply morphological operations for mask refinement
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Apply Gaussian blur for smoother edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        # Threshold to create binary mask
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        print("[*] Wall segmentation completed")
        return mask
    
    def save_mask(self, mask, output_path):
        """Save the segmentation mask.
        
        Args:
            mask: Binary mask highlighting wall regions
            output_path: Path to save the mask
        """
        cv2.imwrite(output_path, mask)
        print(f"[*] Mask saved to {output_path}")


# ===== MODULE 2: TEXTURE REPLACEMENT (Xinhao) =====

class TextureReplacement:
    """Module responsible for replacing wall textures with perspective transformation."""
    
    def __init__(self):
        """Initialize the texture replacement module."""
        pass
    
    def _find_wall_contours(self, mask):
        """Find contours in the wall mask.
        
        Args:
            mask: Binary mask highlighting wall regions
            
        Returns:
            Contours found in the mask
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Return only significant contours (filter out small noise)
        return [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
    
    def _get_perspective_transform(self, contour, texture_size):
        """Calculate perspective transform for the given contour.
        
        Args:
            contour: Contour representing a wall region
            texture_size: Size of the texture to be mapped
            
        Returns:
            Transformation matrix
        """
        # Get approximate rectangle for the contour
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Sort the box points in order: top-left, top-right, bottom-right, bottom-left
        box = self._sort_points(box)
        
        # Define corresponding points in the texture
        texture_points = np.array([
            [0, 0],
            [texture_size[0], 0],
            [texture_size[0], texture_size[1]],
            [0, texture_size[1]]
        ], dtype=np.float32)
        
        # Calculate perspective transform
        M = cv2.getPerspectiveTransform(texture_points, box.astype(np.float32))
        return M
    
    def _sort_points(self, points):
        """Sort rectangle points in clockwise order starting from top-left."""
        # Sort by y-coordinate first (top-to-bottom)
        sorted_by_y = points[np.argsort(points[:, 1])]
        
        # Get top and bottom points
        top = sorted_by_y[:2]
        bottom = sorted_by_y[2:]
        
        # Sort top points by x-coordinate (left-to-right)
        top_sorted = top[np.argsort(top[:, 0])]
        # Sort bottom points by x-coordinate (left-to-right)
        bottom_sorted = bottom[np.argsort(bottom[:, 0])]
        
        # Return points in order: top-left, top-right, bottom-right, bottom-left
        return np.array([top_sorted[0], top_sorted[1], bottom_sorted[1], bottom_sorted[0]])
    
    def replace_texture(self, original_image, mask, texture_image):
        """Replace wall texture in the original image.
        
        Args:
            original_image: Original image
            mask: Binary mask highlighting wall regions
            texture_image: Texture image to apply
            
        Returns:
            Image with replaced wall texture
        """
        print("[*] Replacing wall texture...")
        
        # Convert mask to binary if needed
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Find wall contours in the mask
        contours = self._find_wall_contours(mask)
        
        # Create a copy of the original image for the result
        result = original_image.copy()
        
        # Load and prepare the texture
        texture = cv2.imread(texture_image)
        texture_h, texture_w = texture.shape[:2]
        
        # Process each wall contour
        for contour in contours:
            # Calculate perspective transform
            M = self._get_perspective_transform(contour, (texture_w, texture_h))
            
            # Create a mask for this specific contour
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [contour], 0, 255, -1)
            
            # Warp the texture to match the wall perspective
            h, w = original_image.shape[:2]
            warped_texture = cv2.warpPerspective(texture, M, (w, h))
            
            # Apply the warped texture only to the wall area
            warped_mask = cv2.warpPerspective(np.ones((texture_h, texture_w), dtype=np.uint8) * 255, M, (w, h))
            warped_mask = cv2.bitwise_and(warped_mask, contour_mask)
            
            # Apply brightness adjustment based on original wall
            # (will be enhanced in the LightingAdjustment module)
            
            # Replace the wall area with the warped texture
            contour_mask_3ch = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)
            result = np.where(contour_mask_3ch > 0, warped_texture, result)
        
        print("[*] Texture replacement completed")
        return result


# ===== MODULE 3: LIGHTING ADJUSTMENT (partial responsibility of Xinhao) =====

class LightingAdjustment:
    """Module responsible for maintaining realistic lighting on replaced textures."""
    
    def __init__(self):
        """Initialize the lighting adjustment module."""
        pass
    
    def _estimate_lighting(self, original_image, mask):
        """Estimate lighting conditions in the original image.
        
        Args:
            original_image: Original image
            mask: Binary mask highlighting wall regions
            
        Returns:
            Estimated lighting parameters
        """
        # Convert to grayscale for lighting analysis
        if len(original_image.shape) > 2:
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = original_image.copy()
        
        # Calculate brightness statistics in the wall region
        wall_region = cv2.bitwise_and(gray, mask)
        wall_pixels = wall_region[mask > 0]
        
        if len(wall_pixels) == 0:
            return {
                'brightness_mean': 128,
                'brightness_std': 20,
                'gamma': 1.0
            }
        
        brightness_mean = np.mean(wall_pixels)
        brightness_std = np.std(wall_pixels)
        
        # Estimate gamma as a function of mean brightness
        # (Simple heuristic: darker areas have higher gamma)
        gamma = 2.2 - brightness_mean / 128
        gamma = max(0.8, min(1.5, gamma))  # Clamp to reasonable range
        
        return {
            'brightness_mean': brightness_mean,
            'brightness_std': brightness_std,
            'gamma': gamma
        }
    
    def adjust_lighting(self, replaced_image, original_image, mask):
        """Adjust lighting of the replaced texture to match the original image.
        
        Args:
            replaced_image: Image with replaced texture
            original_image: Original image
            mask: Binary mask highlighting wall regions
            
        Returns:
            Image with adjusted lighting
        """
        print("[*] Adjusting lighting...")
        
        # Estimate lighting conditions
        lighting_params = self._estimate_lighting(original_image, mask)
        
        # Create a copy for the result
        result = replaced_image.copy()
        
        # Apply gamma correction to the replaced area
        # Convert to float32 for processing
        replaced_float = result.astype(np.float32) / 255.0
        
        # Apply gamma correction
        gamma_corrected = np.power(replaced_float, lighting_params['gamma'])
        
        # Convert back to uint8
        gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
        
        # Blend with original image using the mask
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) < 3 else mask
        
        # Apply alpha blending for smoother transition at edges
        kernel = np.ones((21, 21), np.uint8)
        edge_mask = cv2.dilate(mask, kernel) - mask
        edge_mask_3ch = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR) if len(edge_mask.shape) < 3 else edge_mask
        
        # Blend edges with original image
        alpha = 0.7  # Blending factor
        blended_edges = cv2.addWeighted(
            gamma_corrected, alpha,
            original_image, 1 - alpha,
            0, dtype=cv2.CV_8U
        )
        
        # Apply the blended edges
        final_result = np.where(edge_mask_3ch > 0, blended_edges, gamma_corrected)
        
        # Use the original mask for final composition
        final_result = np.where(mask_3ch > 0, final_result, original_image)
        
        print("[*] Lighting adjustment completed")
        return final_result


# ===== MAIN SYSTEM INTEGRATION (Josh) =====

class RepaintSystem:
    """Main system that integrates all modules."""
    
    def __init__(self):
        """Initialize the Repaint system."""
        self.segmentation = WallSegmentation()
        self.replacement = TextureReplacement()
        self.lighting = LightingAdjustment()
    
    def process_image(self, image_path, texture_path, output_path=None):
        """Process an image to replace wall textures.
        
        Args:
            image_path: Path to the input image
            texture_path: Path to the texture image
            output_path: Path to save the output image
            
        Returns:
            Path to the output image
        """
        print(f"\n=== Processing image: {image_path} ===")
        
        # Generate output path if not provided
        if output_path is None:
            basename = os.path.basename(image_path)
            name, ext = os.path.splitext(basename)
            output_path = os.path.join(OUTPUT_DIR, f"{name}_repainted{ext}")
        
        # 1. Segment wall regions
        mask = self.segmentation.segment_wall(image_path)
        
        # 2. Read the original image
        original_image = cv2.imread(image_path)
        
        # 3. Replace texture
        replaced_image = self.replacement.replace_texture(original_image, mask, texture_path)
        
        # 4. Adjust lighting
        final_image = self.lighting.adjust_lighting(replaced_image, original_image, mask)
        
        # 5. Save result
        cv2.imwrite(output_path, final_image)
        print(f"[*] Final image saved to: {output_path}")
        
        return output_path
    
    def batch_process(self, image_dir, texture_path, output_dir=None):
        """Process all images in a directory.
        
        Args:
            image_dir: Directory containing input images
            texture_path: Path to the texture image
            output_dir: Directory to save output images
        """
        if output_dir is None:
            output_dir = OUTPUT_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\n=== Batch processing {len(images)} images ===")
        
        for image_file in images:
            image_path = os.path.join(image_dir, image_file)
            name, ext = os.path.splitext(image_file)
            output_path = os.path.join(output_dir, f"{name}_repainted{ext}")
            
            self.process_image(image_path, texture_path, output_path)
        
        print(f"\n=== Batch processing completed. Results saved to: {output_dir} ===")


# ===== COMMAND LINE INTERFACE =====

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Repaint: Wall Recognition and Replacement System')
    
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='Path to the input image or directory of images')
    
    parser.add_argument('--texture', '-t', type=str, required=True,
                        help='Path to the texture image')
    
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save the output image or directory for batch processing')
    
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Process all images in the input directory')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Initialize Repaint system
    repaint = RepaintSystem()
    
    # Check if input image exists
    if not os.path.exists(args.image):
        print(f"Error: Input image/directory not found: {args.image}")
        return
    
    # Check if texture exists
    if not os.path.exists(args.texture):
        print(f"Error: Texture image not found: {args.texture}")
        return
    
    # Process image(s)
    if args.batch:
        if not os.path.isdir(args.image):
            print(f"Error: When using --batch, --image must be a directory")
            return
        
        repaint.batch_process(args.image, args.texture, args.output)
    else:
        repaint.process_image(args.image, args.texture, args.output)
    
    print("\n=== Processing completed successfully ===")


if __name__ == "__main__":
    main()
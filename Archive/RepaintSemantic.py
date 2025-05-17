"""
RepaintSemantic: Enhanced Wall Recognition and Replacement System using Semantic Segmentation

This application uses pre-trained semantic segmentation models to accurately detect wall areas
in interior images, then applies texture replacement with realistic lighting adjustments.

Team members:
- Lihua: Image segmentation module (semantic segmentation)
- Xinhao: Texture replacement and composition module
- Josh: System integration and testing
"""

import os
import sys
import numpy as np
import cv2
import torch
import argparse
import requests
import zipfile
import io
from PIL import Image
import torchvision.transforms as transforms

# Define the project structure
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "semantic_models")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_semantic")

# Create directories if they don't exist
for directory in [MODELS_DIR, SAMPLES_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# ===== MODULE 1: SEMANTIC SEGMENTATION FOR WALL DETECTION (Lihua) =====

class SemanticWallSegmentation:
    """使用语义分割模型进行墙面识别"""
    
    def __init__(self, model_type="deeplabv3_resnet101"):
        """
        初始化语义分割模型
        
        Args:
            model_type: 模型类型，可选值: 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large'
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Using device: {self.device}")
        
        # 检查模型文件是否存在，如果不存在则下载
        self._ensure_model_available(model_type)
        
        # 加载语义分割模型
        self.model_type = model_type
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # ADE20K数据集的类别映射（墙壁通常是类别：1-墙, 4-墙纸）
        self.wall_classes = [1, 4, 9]  # 墙和墙纸的类别ID
        
        # 定义图像预处理转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _ensure_model_available(self, model_type):
        """确保模型文件可用，如果不存在则下载"""
        print(f"[*] Checking if model '{model_type}' is available...")
        
        # 使用torchvision提供的预训练模型，无需手动下载
        pass
    
    def _load_model(self):
        """加载语义分割模型"""
        print(f"[*] Loading semantic segmentation model: {self.model_type}")
        
        try:
            # 使用torchvision提供的预训练模型
            if self.model_type == "deeplabv3_resnet101":
                from torchvision.models.segmentation import deeplabv3_resnet101
                model = deeplabv3_resnet101(pretrained=True)
            elif self.model_type == "deeplabv3_mobilenet_v3_large":
                from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
                model = deeplabv3_mobilenet_v3_large(pretrained=True)
            elif self.model_type == "lraspp_mobilenet_v3_large":
                from torchvision.models.segmentation import lraspp_mobilenet_v3_large
                model = lraspp_mobilenet_v3_large(pretrained=True)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
            print(f"[*] Model loaded successfully!")
            return model
            
        except Exception as e:
            print(f"[!] Error loading model: {str(e)}")
            raise
    
    def segment_wall(self, image_path):
        """
        使用语义分割识别墙面区域
        
        Args:
            image_path: 输入图像路径
            
        Returns:
            墙面区域的二值掩码
        """
        print(f"[*] Segmenting wall with semantic segmentation on: {image_path}")
        
        # 读取图像
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        
        # 预处理图像
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 运行模型推理
        with torch.no_grad():
            output = self.model(input_tensor)["out"][0]
            output = torch.softmax(output, dim=0)
            
            # 创建一个包含所有墙类别的掩码
            wall_mask = torch.zeros_like(output[0])
            for wall_class in self.wall_classes:
                if wall_class < output.shape[0]:  # 确保类别ID在模型输出范围内
                    wall_mask += output[wall_class]
            
            # 将掩码转换为二值图像
            wall_mask = wall_mask > 0.5
            wall_mask = wall_mask.cpu().numpy().astype(np.uint8) * 255
            
            # 调整回原始图像尺寸
            wall_mask = cv2.resize(wall_mask, (original_size[0], original_size[1]))
            
        # 应用形态学操作，平滑边缘并填充小洞
        kernel = np.ones((10, 10), np.uint8)
        refined_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
        
        # 创建可视化结果
        self._create_visualization(image_path, wall_mask)
        
        print("[*] Wall segmentation completed with semantic segmentation!")
        return refined_mask
    
    def _create_visualization(self, image_path, mask):
        """创建分割结果的可视化"""
        try:
            # 创建输出目录
            vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # 读取原始图像
            original_image = cv2.imread(image_path)
            
            # 获取文件名（不含扩展名）
            basename = os.path.basename(image_path)
            name, ext = os.path.splitext(basename)
            
            # 保存掩码
            mask_path = os.path.join(vis_dir, f"{name}_semantic_mask.png")
            cv2.imwrite(mask_path, mask)
            
            # 创建彩色掩码叠加在原图上
            colored_mask = np.zeros_like(original_image)
            colored_mask[mask > 128] = [0, 0, 255]  # 红色表示墙
            
            # 半透明叠加
            overlay = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)
            overlay_path = os.path.join(vis_dir, f"{name}_semantic_overlay.png")
            cv2.imwrite(overlay_path, overlay)
            
            # 绘制轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = original_image.copy()
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
            contour_path = os.path.join(vis_dir, f"{name}_semantic_contours.png")
            cv2.imwrite(contour_path, contour_image)
            
            print(f"[*] Visualization saved: {vis_dir}")
        except Exception as e:
            print(f"[!] Error creating visualization: {str(e)}")

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
        box = box.astype(np.intp)  
        
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
    
    def __init__(self, model_type="deeplabv3_resnet101"):
        """Initialize the Repaint system."""
        self.segmentation = SemanticWallSegmentation(model_type)
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
        
        # 6. Create comparison visualization
        vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        basename = os.path.basename(image_path)
        name, ext = os.path.splitext(basename)
        
        # Create side-by-side comparison
        h, w = original_image.shape[:2]
        comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
        comparison[:, :w] = original_image
        comparison[:, w:] = final_image
        comparison_path = os.path.join(vis_dir, f"{name}_comparison.png")
        cv2.imwrite(comparison_path, comparison)
        
        print(f"[*] Final image saved to: {output_path}")
        print(f"[*] Comparison visualization saved to: {comparison_path}")
        
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
    parser = argparse.ArgumentParser(description='RepaintSemantic: Wall Recognition and Replacement System using Semantic Segmentation')
    
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='Path to the input image or directory of images')
    
    parser.add_argument('--texture', '-t', type=str, required=True,
                        help='Path to the texture image')
    
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save the output image or directory for batch processing')
    
    parser.add_argument('--model', '-m', type=str, default="deeplabv3_resnet101",
                        choices=["deeplabv3_resnet101", "deeplabv3_mobilenet_v3_large", "lraspp_mobilenet_v3_large"],
                        help='Type of semantic segmentation model to use')
    
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Process all images in the input directory')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Initialize Repaint system
    repaint = RepaintSystem(model_type=args.model)
    
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
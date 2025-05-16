"""
RepaintCustom: 墙面识别与替换系统（基于用户初选+算法细化）- 第三部分

此部分包含主系统集成、交互式墙面选择功能和命令行界面。
"""

import os
import sys
import numpy as np
import cv2
import argparse
import time
from datetime import datetime

# 导入自定义模块
from selection import CustomWallDetector, user_wall_selection
from texture import TextureReplacement, LightingAdjustment

# 定义项目结构
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_custom")

# 创建目录（如果不存在）
for directory in [SAMPLES_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# ===== 主系统集成 =====

class RepaintSystem:
    """集成所有模块的主系统，支持用户选择墙面区域"""
    
    def __init__(self, color_tolerance=15, shadow_tolerance=30, min_region_size=1000):
        """初始化Repaint系统"""
        self.wall_detector = CustomWallDetector(
            color_tolerance=color_tolerance,
            shadow_tolerance=shadow_tolerance,
            min_region_size=min_region_size
        )
        self.replacement = TextureReplacement()
        self.lighting = LightingAdjustment()
    
    def process_image(self, image_path, texture_path, user_selection=None, output_path=None):
        """
        处理图像以替换墙面纹理，支持用户预选择墙面区域
        
        Args:
            image_path: 输入图像路径
            texture_path: 纹理图像路径
            user_selection: 用户选择参数，可以是以下格式之一:
                - None: 不使用用户选择，算法自动检测
                - 'auto': 不使用用户选择，算法自动检测
                - {'type': 'rect', 'rect': (x, y, w, h)}: 矩形选择
                - {'type': 'points', 'points': [(x1,y1), (x2,y2), ...]}: 多边形选择
            output_path: 保存输出图像的路径
            
        Returns:
            输出图像的路径
        """
        print(f"\n=== 处理图像: {image_path} ===")
        
        # 若未提供输出路径则生成
        if output_path is None:
            basename = os.path.basename(image_path)
            name, ext = os.path.splitext(basename)
            output_path = os.path.join(OUTPUT_DIR, f"{name}_repainted{ext}")
        
        # 1. 用户选择墙面区域（如果提供）
        user_mask = None
        if user_selection is not None and user_selection != 'auto':
            selection_type = user_selection.get('type', 'rect')
            
            if selection_type == 'rect':
                user_mask = user_wall_selection(
                    image_path, 
                    selection_type='rect', 
                    rect=user_selection.get('rect')
                )
            elif selection_type == 'points':
                user_mask = user_wall_selection(
                    image_path, 
                    selection_type='points', 
                    points=user_selection.get('points')
                )
            
            print(f"[*] 用户已选择墙面区域")
        
        # 2. 检测墙面区域（在用户选择区域内进行细化）
        mask = self.wall_detector.detect_wall(image_path, user_mask)
        
        # 3. 读取原始图像
        original_image = cv2.imread(image_path)
        
        # 4. 替换纹理
        replaced_image = self.replacement.replace_texture(original_image, mask, texture_path)
        
        # 5. 调整光照
        final_image = self.lighting.adjust_lighting(replaced_image, original_image, mask)
        
        # 6. 保存结果
        cv2.imwrite(output_path, final_image)
        
        # 7. 创建对比可视化
        vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        basename = os.path.basename(image_path)
        name, ext = os.path.splitext(basename)
        
        # 创建并排对比
        h, w = original_image.shape[:2]
        comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
        comparison[:, :w] = original_image
        comparison[:, w:] = final_image
        comparison_path = os.path.join(vis_dir, f"{name}_comparison.png")
        cv2.imwrite(comparison_path, comparison)
        
        print(f"[*] 最终图像已保存至: {output_path}")
        print(f"[*] 对比可视化已保存至: {comparison_path}")
        
        return output_path
    
    def batch_process(self, image_dir, texture_path, user_selections=None, output_dir=None):
        """
        批量处理目录中的所有图像，支持为每个图像指定用户选择
        
        Args:
            image_dir: 包含输入图像的目录
            texture_path: 纹理图像路径
            user_selections: 用户选择字典 {image_name: selection_params}
                selection_params 格式同 process_image 中的 user_selection 参数
            output_dir: 保存输出图像的目录
        """
        if output_dir is None:
            output_dir = OUTPUT_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\n=== 批量处理 {len(images)} 张图像 ===")
        
        for image_file in images:
            image_path = os.path.join(image_dir, image_file)
            name, ext = os.path.splitext(image_file)
            output_path = os.path.join(output_dir, f"{name}_repainted{ext}")
            
            # 获取当前图像的用户选择（如果有）
            user_selection = None
            if user_selections is not None:
                user_selection = user_selections.get(image_file) or user_selections.get(name)
            
            self.process_image(image_path, texture_path, user_selection, output_path)
        
        print(f"\n=== 批量处理完成。结果已保存至: {output_dir} ===")


# ===== 交互式墙面选择功能 =====

def interactive_wall_selection(image_path):
    """
    提供交互式界面让用户选择墙面区域
    
    Args:
        image_path: 输入图像路径
        
    Returns:
        用户选择的区域掩码和选择参数
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 存储用户选择的点
    points = []
    # 存储是否完成选择
    selection_done = False
    # 存储临时图像（用于显示）
    temp_img = image.copy()
    
    # 鼠标回调函数
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, selection_done, temp_img
        
        # 左键点击添加点
        if event == cv2.EVENT_LBUTTONDOWN and not selection_done:
            points.append((x, y))
            # 在图像上显示点
            cv2.circle(temp_img, (x, y), 5, (0, 255, 0), -1)
            # 如果有多个点，绘制线段
            if len(points) > 1:
                cv2.line(temp_img, points[-2], points[-1], (0, 255, 0), 2)
            cv2.imshow("选择墙面区域", temp_img)
        
        # 右键点击完成选择
        elif event == cv2.EVENT_RBUTTONDOWN and len(points) > 2:
            # 连接最后一个点和第一个点
            cv2.line(temp_img, points[-1], points[0], (0, 255, 0), 2)
            selection_done = True
            cv2.imshow("选择墙面区域", temp_img)
    
    # 创建窗口并设置鼠标回调
    cv2.namedWindow("选择墙面区域")
    cv2.setMouseCallback("选择墙面区域", mouse_callback)
    
    # 显示初始图像
    cv2.imshow("选择墙面区域", image)
    print("使用鼠标左键在图像上选择墙面区域的轮廓点。")
    print("完成选择后，点击鼠标右键。")
    print("按'r'键重置选择，按'ESC'键取消。")
    
    # 事件循环
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # 按ESC退出
        if key == 27:
            points = []
            break
        
        # 按r重置选择
        if key == ord('r'):
            points = []
            selection_done = False
            temp_img = image.copy()
            cv2.imshow("选择墙面区域", temp_img)
        
        # 按Enter完成选择
        if (key == 13 or selection_done) and len(points) > 2:
            break
    
    cv2.destroyAllWindows()
    
    # 如果用户取消选择
    if not points:
        print("用户取消了选择。")
        return None, None
    
    # 创建掩码
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    
    # 创建选择参数
    selection_params = {
        'type': 'points',
        'points': points
    }
    
    print(f"用户已选择 {len(points)} 个点定义墙面区域。")
    
    return mask, selection_params


# ===== 命令行界面 =====

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RepaintCustom: 墙面识别与替换系统（基于用户初选+算法细化）')
    
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='输入图像路径或图像目录')
    
    parser.add_argument('--texture', '-t', type=str, required=True,
                        help='纹理图像路径')
    
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='保存输出图像的路径或批处理的输出目录')
    
    parser.add_argument('--batch', '-b', action='store_true',
                        help='批量处理输入目录中的所有图像')
    
    parser.add_argument('--color-tolerance', type=int, default=15,
                        help='颜色相似度容差 (RGB差异,default=15)')
    
    parser.add_argument('--shadow-tolerance', type=int, default=30,
                        help='阴影/光照变化容差, default=30')
    
    parser.add_argument('--min-region-size', type=int, default=1000,
                        help='最小区域大小（以像素计, default=1000）')
    
    parser.add_argument('--interactive', '-int', action='store_true',
                        help='使用交互式界面选择墙面区域')
    
    parser.add_argument('--rect', type=str, default=None,
                        help='矩形选择区域 "x,y,w,h"，例如 "100,200,300,400"')
    
    parser.add_argument('--auto', action='store_true',
                        help='使用全自动墙面检测（不使用用户选择）')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 初始化Repaint系统
    repaint = RepaintSystem(
        color_tolerance=args.color_tolerance,
        shadow_tolerance=args.shadow_tolerance,
        min_region_size=args.min_region_size
    )
    
    # 检查输入参数
    if not os.path.exists(args.image):
        print(f"错误: 未找到输入图像/目录: {args.image}")
        return
    
    if not os.path.exists(args.texture):
        print(f"错误: 未找到纹理图像: {args.texture}")
        return
    
    # 处理用户选择参数
    user_selection = None
    
    # 自动模式
    if args.auto:
        user_selection = 'auto'
    
    # 矩形选择
    elif args.rect is not None:
        try:
            rect = [int(x) for x in args.rect.split(',')]
            if len(rect) != 4:
                raise ValueError("矩形参数必须是四个整数: x,y,w,h")
            user_selection = {'type': 'rect', 'rect': tuple(rect)}
        except Exception as e:
            print(f"错误: 无效的矩形参数 - {str(e)}")
            return
    
    # 处理图像
    if args.batch:
        if not os.path.isdir(args.image):
            print(f"错误: 使用--batch时，--image必须是目录")
            return
        
        # 批处理模式不支持交互式选择
        if args.interactive:
            print("警告: 批处理模式不支持交互式选择。将对所有图像使用相同的选择参数。")
        
        repaint.batch_process(args.image, args.texture, 
                            user_selections={None: user_selection}, 
                            output_dir=args.output)
    else:
        # 单图像处理
        if args.interactive:
            print("\n=== 交互式墙面选择 ===")
            _, user_selection = interactive_wall_selection(args.image)
            if user_selection is None:
                print("未选择墙面区域，将使用全自动检测。")
                user_selection = 'auto'
        
        repaint.process_image(args.image, args.texture, user_selection, args.output)
    
    print("\n=== 处理成功完成 ===")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\n总运行时间: {elapsed_time:.2f} 秒")
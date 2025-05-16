"""
RepaintCustom: 墙面识别与替换系统（基于用户初选+算法细化）- 第一部分

此部分包含项目结构定义、用户墙面选择功能和自定义墙面识别模块。
"""

import os
import sys
import numpy as np
import cv2
import argparse
from PIL import Image
import time
from datetime import datetime

# 定义项目结构
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_custom")

# 创建目录（如果不存在）
for directory in [SAMPLES_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# ===== 用户初选墙面功能 =====

def user_wall_selection(image_path, selection_type='rect', points=None, rect=None):
    """
    用户初步选择墙面区域
    
    Args:
        image_path: 图像路径
        selection_type: 选择类型 ('rect'表示矩形, 'points'表示多边形)
        points: 多边形的点列表 [(x1,y1), (x2,y2), ...]，仅当selection_type='points'时使用
        rect: 矩形区域 (x, y, width, height)，仅当selection_type='rect'时使用
        
    Returns:
        用户选择的区域掩码
    """
    print(f"[*] 用户初选墙面区域: {image_path}")
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if selection_type == 'rect':
        # 使用矩形选择
        if rect is None:
            # 默认选择图像中心的区域
            x = width // 4
            y = height // 4
            w = width // 2
            h = height // 2
            rect = (x, y, w, h)
        
        x, y, w, h = rect
        # 确保矩形在图像范围内
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        
        # 填充选定的矩形区域
        mask[y:y+h, x:x+w] = 255
        print(f"[*] 已选择矩形区域: x={x}, y={y}, 宽={w}, 高={h}")
        
    elif selection_type == 'points':
        # 使用多边形选择
        if points is None or len(points) < 3:
            # 默认选择图像中心的一个三角形
            cx, cy = width // 2, height // 2
            size = min(width, height) // 4
            points = [
                (cx, cy - size),
                (cx - size, cy + size),
                (cx + size, cy + size)
            ]
        
        # 将点转换为numpy数组格式
        pts = np.array(points, dtype=np.int32)
        # 绘制填充多边形
        cv2.fillPoly(mask, [pts], 255)
        print(f"[*] 已选择多边形区域: {len(points)}个点")
    
    # 创建可视化结果
    vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 获取文件名（不含扩展名）
    basename = os.path.basename(image_path)
    name, ext = os.path.splitext(basename)
    
    # 保存掩码
    mask_path = os.path.join(vis_dir, f"{name}_user_selection.png")
    cv2.imwrite(mask_path, mask)
    
    # 创建彩色掩码叠加在原图上
    overlay = image.copy()
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [0, 0, 255]  # 红色表示用户选择
    
    # 半透明叠加
    alpha = 0.3
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    
    # 绘制边界线
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    
    # 保存可视化结果
    overlay_path = os.path.join(vis_dir, f"{name}_user_selection_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    
    print(f"[*] 用户选择可视化已保存至: {overlay_path}")
    return mask

# ===== 自定义墙面识别算法 =====

class CustomWallDetector:
    """使用自定义区域增长和颜色分析算法进行墙面识别"""
    
    def __init__(self, color_tolerance=15, shadow_tolerance=30, min_region_size=1000):
        """
        初始化墙面识别器
        
        Args:
            color_tolerance: 颜色相似度容差 (RGB差异)
            shadow_tolerance: 阴影/光照变化容差
            min_region_size: 最小区域大小（以像素计）
        """
        self.color_tolerance = color_tolerance
        self.shadow_tolerance = shadow_tolerance
        self.min_region_size = min_region_size
    
    def _is_similar_color(self, color1, color2):
        """
        判断两个颜色是否相似
        通过分析RGB差异和亮度差异独立判断
        
        Args:
            color1: BGR格式的第一个颜色
            color2: BGR格式的第二个颜色
            
        Returns:
            是否相似的布尔值
        """
        # 颜色差异 (RGB分量)
        color_diff = np.abs(color1.astype(np.int32) - color2.astype(np.int32))
        max_color_diff = np.max(color_diff)
        
        # 亮度差异
        brightness1 = np.sum(color1) / 3
        brightness2 = np.sum(color2) / 3
        brightness_diff = abs(brightness1 - brightness2)
        
        # 条件1: 所有RGB分量差异小于容差
        color_similar = max_color_diff <= self.color_tolerance
        
        # 条件2: 亮度差异在阴影容差内
        brightness_similar = brightness_diff <= self.shadow_tolerance
        
        # 条件3: 色调相似（即使亮度差异大）
        # 归一化颜色来比较色调
        if brightness1 > 0 and brightness2 > 0:
            norm_color1 = color1 / brightness1
            norm_color2 = color2 / brightness2
            hue_diff = np.abs(norm_color1 - norm_color2)
            hue_similar = np.max(hue_diff) < 0.3  # 相对宽松的色调相似度阈值
        else:
            hue_similar = False
        
        # 满足颜色相似或（亮度相似且色调相似）
        return color_similar or (brightness_similar and hue_similar)
    
    def _region_growing(self, image, seed_points, user_mask=None):
        """
        从种子点开始的多点区域增长算法，限制在用户选择的区域内
        
        Args:
            image: 输入图像（BGR格式）
            seed_points: 种子点列表 [(x, y), ...]
            user_mask: 用户选择的区域掩码（可选）
            
        Returns:
            区域掩码（二值图像）
        """
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        visited = np.zeros((height, width), dtype=bool)
        
        # 过滤用户选择区域外的种子点
        if user_mask is not None:
            seed_points = [(x, y) for x, y in seed_points if user_mask[y, x] > 0]
            if not seed_points:
                print("[!] 警告: 所有种子点都在用户选择区域外。使用用户选择区域中的随机点作为种子。")
                y_coords, x_coords = np.where(user_mask > 0)
                if len(y_coords) > 0:
                    # 随机选择用户区域内的10个点
                    indices = np.random.choice(len(y_coords), min(10, len(y_coords)), replace=False)
                    seed_points = [(x_coords[i], y_coords[i]) for i in indices]
        
        # 准备队列
        queue = list(seed_points)
        
        # 设置邻域探索方向：上、右、下、左、四个对角线
        neighbors = [
            (0, -1), (1, 0), (0, 1), (-1, 0),
            (-1, -1), (1, -1), (1, 1), (-1, 1)
        ]
        
        print(f"[*] 开始区域增长算法，使用 {len(seed_points)} 个种子点...")
        start_time = time.time()
        
        # 对于每个种子点
        for seed_x, seed_y in seed_points:
            if visited[seed_y, seed_x]:  # 如果已访问，跳过
                continue
                
            # 获取种子点颜色
            seed_color = image[seed_y, seed_x]
            
            # 当前区域增长的队列
            current_queue = [(seed_x, seed_y)]
            current_region = []
            
            # 区域增长过程
            while current_queue:
                x, y = current_queue.pop(0)
                
                # 如果已访问或在用户选择区域外，跳过
                if visited[y, x] or (user_mask is not None and user_mask[y, x] == 0):
                    continue
                
                # 标记为已访问
                visited[y, x] = True
                
                # 获取当前点颜色
                current_color = image[y, x]
                
                # 检查与种子点的颜色相似度
                if self._is_similar_color(seed_color, current_color):
                    # 添加到当前区域
                    current_region.append((x, y))
                    mask[y, x] = 255
                    
                    # 检查邻居
                    for dx, dy in neighbors:
                        nx, ny = x + dx, y + dy
                        
                        # 检查边界和用户选择区域
                        valid_neighbor = (0 <= nx < width and 0 <= ny < height and 
                                         not visited[ny, nx] and 
                                         (user_mask is None or user_mask[ny, nx] > 0))
                        
                        if valid_neighbor:
                            neighbor_color = image[ny, nx]
                            
                            # 检查邻居与当前点的颜色相似度
                            if self._is_similar_color(current_color, neighbor_color):
                                current_queue.append((nx, ny))
            
            # 如果当前区域太小，从掩码中移除
            if len(current_region) < self.min_region_size:
                for x, y in current_region:
                    mask[y, x] = 0
        
        elapsed_time = time.time() - start_time
        print(f"[*] 区域增长算法完成，耗时 {elapsed_time:.2f} 秒")
        
        return mask
    
    def _select_seed_points(self, image, user_mask=None):
        """
        智能选择种子点，限制在用户选择区域内
        
        Args:
            image: 输入图像
            user_mask: 用户选择的区域掩码（可选）
            
        Returns:
            种子点列表 [(x, y), ...]
        """
        height, width = image.shape[:2]
        seed_points = []
        
        # 基于网格的种子点选择
        grid_step = min(height, width) // 15  # 较密集的网格
        
        # 如果有用户掩码，在用户选择区域内选择种子点
        if user_mask is not None:
            # 获取用户选择区域的边界
            y_coords, x_coords = np.where(user_mask > 0)
            if len(y_coords) == 0:
                return []  # 用户选择区域为空
                
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            
            # 在用户选择区域内创建网格
            for y in range(min_y, max_y, grid_step):
                for x in range(min_x, max_x, grid_step):
                    if 0 <= x < width and 0 <= y < height and user_mask[y, x] > 0:
                        seed_points.append((x, y))
        else:
            # 无用户掩码，使用全图网格
            border = int(min(height, width) * 0.05)  # 较小边界
            for y in range(border, height - border, grid_step):
                for x in range(border, width - border, grid_step):
                    seed_points.append((x, y))
        
        # 避开边缘区域
        edges = cv2.Canny(image, 100, 200)
        dilated_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        
        # 过滤掉边缘附近的种子点
        seed_points = [(x, y) for x, y in seed_points if dilated_edges[y, x] == 0]
        
        # 基于颜色直方图添加额外种子点
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 应用用户掩码（如果有）
        if user_mask is not None:
            hsv_masked = hsv.copy()
            mask_3d = np.stack([user_mask] * 3, axis=2) > 0
            # 只考虑用户选择区域内的HSV值
            hsv_masked = np.where(mask_3d, hsv, 0)
            hsv = hsv_masked
        
        # 计算色调和饱和度的直方图
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        
        # 找出主要色调和饱和度范围
        h_peaks = np.where(h_hist > 0.8 * h_hist.max())[0]
        s_peaks = np.where(s_hist > 0.7 * s_hist.max())[0]
        
        # 在主要颜色区域添加种子点
        for h_range in h_peaks:
            for s_range in s_peaks:
                # 创建颜色范围掩码
                in_range_mask = cv2.inRange(
                    hsv, 
                    np.array([h_range, s_range, 50]), 
                    np.array([h_range + 1, s_range + 1, 255])
                )
                
                # 应用用户掩码（如果有）
                if user_mask is not None:
                    in_range_mask = cv2.bitwise_and(in_range_mask, user_mask)
                
                if np.sum(in_range_mask) > 100:
                    y_coords, x_coords = np.where(in_range_mask > 0)
                    if len(y_coords) > 0:
                        # 从区域随机选择点
                        idx = np.random.randint(0, len(y_coords))
                        seed_points.append((x_coords[idx], y_coords[idx]))
        
        print(f"[*] 已选择 {len(seed_points)} 个种子点")
        return seed_points
    
    def _refine_mask(self, mask, image, user_mask=None):
        """
        对掩码进行后处理优化，限制在用户选择区域内
        
        Args:
            mask: 初始掩码
            image: 原始图像
            user_mask: 用户选择的区域掩码（可选）
            
        Returns:
            优化后的掩码
        """
        # 应用用户选择区域约束
        if user_mask is not None:
            mask = cv2.bitwise_and(mask, user_mask)
        
        # 1. 应用形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 2. 移除小区域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < self.min_region_size:
                cv2.drawContours(mask, [contour], -1, 0, -1)
        
        # 3. 平滑边缘
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 4. 使用边缘改进边界
        edges = cv2.Canny(image, 50, 150)
        dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        # 在用户选择区域内使用边缘信息
        if user_mask is not None:
            dilated_edges = cv2.bitwise_and(dilated_edges, user_mask)
        
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(dilated_edges))
        
        return mask
    
    def _create_visualization(self, image_path, mask, user_mask=None):
        """创建分割结果的可视化，显示用户选择和算法细化的区域"""
        try:
            vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            original_image = cv2.imread(image_path)
            basename = os.path.basename(image_path)
            name, ext = os.path.splitext(basename)
            
            # 保存算法掩码
            mask_path = os.path.join(vis_dir, f"{name}_algorithm_mask.png")
            cv2.imwrite(mask_path, mask)
            
            # 创建可视化结果
            visualization = original_image.copy()
            
            # 如果有用户掩码，先显示用户选择区域（淡蓝色）
            if user_mask is not None:
                user_overlay = np.zeros_like(original_image)
                user_overlay[user_mask > 0] = [255, 200, 100]  # 淡蓝色
                visualization = cv2.addWeighted(visualization, 0.85, user_overlay, 0.15, 0)
                
                # 绘制用户选择边界
                user_contours, _ = cv2.findContours(user_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(visualization, user_contours, -1, (100, 200, 255), 2)
            
            # 显示算法检测区域（红色）
            algo_overlay = np.zeros_like(original_image)
            algo_overlay[mask > 0] = [0, 0, 255]  # 红色
            visualization = cv2.addWeighted(visualization, 0.7, algo_overlay, 0.3, 0)
            
            # 绘制算法检测边界
            algo_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(visualization, algo_contours, -1, (0, 255, 0), 2)
            
            # 添加文字说明
            font = cv2.FONT_HERSHEY_SIMPLEX
            if user_mask is not None:
                cv2.putText(visualization, "User Selection (Blue)", (10, 30), font, 0.7, (100, 200, 255), 2)
            cv2.putText(visualization, "Algorithm Detection (Red/Green)", (10, 60), font, 0.7, (0, 255, 0), 2)
            
            # 保存最终可视化
            vis_path = os.path.join(vis_dir, f"{name}_detection_visualization.png")
            cv2.imwrite(vis_path, visualization)
            
            print(f"[*] 可视化结果已保存至: {vis_path}")
        except Exception as e:
            print(f"[!] 创建可视化时出错: {str(e)}")
    
    def detect_wall(self, image_path, user_mask=None):
        """
        检测图像中的墙面区域，限制在用户选择区域内
        
        Args:
            image_path: 输入图像路径
            user_mask: 用户选择的区域掩码（可选）
            
        Returns:
            墙面区域的二值掩码
        """
        print(f"[*] 使用自定义算法检测墙面: {image_path}")
        
        # 检查是否为纯墙面图像
        filename = os.path.basename(image_path).lower()
        is_wall_only = 'wall' in filename or 'brick' in filename
        
        # 如果是纯墙面图像且没有用户选择，使用整图作为掩码
        if is_wall_only and user_mask is None:
            print(f"[*] 检测到纯墙面图像，使用整图作为墙面")
            image = cv2.imread(image_path)
            wall_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
            self._create_visualization(image_path, wall_mask)
            return wall_mask
        
        # 如果有用户掩码，并且是纯墙面图像，直接使用用户掩码
        if is_wall_only and user_mask is not None:
            print(f"[*] 检测到纯墙面图像，使用用户选择区域作为墙面")
            self._create_visualization(image_path, user_mask, user_mask)
            return user_mask
        
        # 常规处理：读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 预处理：平滑图像
        smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)
        
        # 选择种子点（在用户选择区域内）
        seed_points = self._select_seed_points(smoothed_image, user_mask)
        
        # 区域增长算法（受限于用户选择区域）
        initial_mask = self._region_growing(smoothed_image, seed_points, user_mask)
        
        # 优化掩码（受限于用户选择区域）
        refined_mask = self._refine_mask(initial_mask, image, user_mask)
        
        # 创建可视化结果
        self._create_visualization(image_path, refined_mask, user_mask)
        
        print(f"[*] 墙面检测完成: {image_path}")
        return refined_mask
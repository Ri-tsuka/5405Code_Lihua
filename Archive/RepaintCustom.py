"""
RepaintCustom: 墙面识别与替换系统 (自定义算法版)

此应用使用自定义的区域增长和颜色相似度分析来识别墙面区域，
然后应用纹理替换并进行真实光照调整。

团队成员:
- Lihua: 墙面识别模块 (自定义算法)
- Xinhao: 纹理替换和合成模块
- Josh: 系统集成和测试
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

# ===== 模块1: 自定义墙面识别算法 (Lihua) =====

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
    
    def _region_growing(self, image, seed_points):
        """
        从种子点开始的多点区域增长算法
        
        Args:
            image: 输入图像（BGR格式）
            seed_points: 种子点列表 [(x, y), ...]
            
        Returns:
            区域掩码（二值图像）
        """
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        visited = np.zeros((height, width), dtype=bool)
        
        # 准备队列
        queue = list(seed_points)
        
        # 设置邻域探索方向：上、右、下、左、四个对角线
        neighbors = [
            (0, -1), (1, 0), (0, 1), (-1, 0),
            (-1, -1), (1, -1), (1, 1), (-1, 1)
        ]
        
        print(f"[*] Starting region growing from {len(seed_points)} seed points...")
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
                
                # 如果已访问，跳过
                if visited[y, x]:
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
                        
                        # 检查边界
                        if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                            neighbor_color = image[ny, nx]
                            
                            # 检查邻居与当前点的颜色相似度
                            if self._is_similar_color(current_color, neighbor_color):
                                current_queue.append((nx, ny))
            
            # 如果当前区域太小，从掩码中移除
            if len(current_region) < self.min_region_size:
                for x, y in current_region:
                    mask[y, x] = 0
        
        elapsed_time = time.time() - start_time
        print(f"[*] Region growing completed in {elapsed_time:.2f} seconds")
        
        return mask
    
    def _select_seed_points(self, image):
        """
        智能选择种子点
        使用均匀网格和边缘检测的组合来选择可能的墙面区域种子点
        
        Args:
            image: 输入图像
            
        Returns:
            种子点列表 [(x, y), ...]
        """
        height, width = image.shape[:2]
        seed_points = []
        
        # 策略1：基于网格的种子点，跳过图像边缘
        grid_step = min(height, width) // 10  # 网格步长
        border = int(min(height, width) * 0.1)  # 边界大小
        
        for y in range(border, height - border, grid_step):
            for x in range(border, width - border, grid_step):
                seed_points.append((x, y))
        
        # 策略2：避开边缘区域的种子点
        edges = cv2.Canny(image, 100, 200)
        dilated_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        
        # 过滤掉边缘附近的种子点
        seed_points = [(x, y) for x, y in seed_points if dilated_edges[y, x] == 0]
        
        # 策略3：加入基于颜色直方图的种子点
        # 将图像转换为HSV以便更好地分析颜色
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 计算色调和饱和度的直方图
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        
        # 找出主要色调和饱和度范围（可能对应墙面）
        h_peaks = np.where(h_hist > 0.8 * h_hist.max())[0]
        s_peaks = np.where(s_hist > 0.7 * s_hist.max())[0]
        
        # 在每个主要颜色区域添加种子点
        for h_range in h_peaks:
            for s_range in s_peaks:
                # 寻找符合该色调和饱和度范围的像素
                in_range_mask = cv2.inRange(
                    hsv, 
                    np.array([h_range, s_range, 50]), 
                    np.array([h_range + 1, s_range + 1, 255])
                )
                
                if np.sum(in_range_mask) > 100:  # 如果有足够的像素
                    y_coords, x_coords = np.where(in_range_mask > 0)
                    if len(y_coords) > 0:
                        # 从该区域随机选择一个点作为种子
                        idx = np.random.randint(0, len(y_coords))
                        seed_points.append((x_coords[idx], y_coords[idx]))
        
        print(f"[*] Selected {len(seed_points)} seed points")
        return seed_points
    
    def _refine_mask(self, mask, image):
        """
        对掩码进行后处理优化
        
        Args:
            mask: 初始掩码
            image: 原始图像
            
        Returns:
            优化后的掩码
        """
        # 1. 应用形态学操作来填充小洞
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 2. 移除小的孤立区域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < self.min_region_size:
                cv2.drawContours(mask, [contour], -1, 0, -1)  # 填充小区域为0
        
        # 3. 平滑边缘
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 4. 基于边缘检测优化掩码边界
        edges = cv2.Canny(image, 50, 150)
        dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        # 使用边缘作为区域分隔
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(dilated_edges))
        
        return mask
    
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
            mask_path = os.path.join(vis_dir, f"{name}_custom_mask.png")
            cv2.imwrite(mask_path, mask)
            
            # 创建彩色掩码叠加在原图上
            colored_mask = np.zeros_like(original_image)
            colored_mask[mask > 128] = [0, 0, 255]  # 红色表示墙
            
            # 半透明叠加
            overlay = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)
            overlay_path = os.path.join(vis_dir, f"{name}_custom_overlay.png")
            cv2.imwrite(overlay_path, overlay)
            
            # 绘制轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = original_image.copy()
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
            contour_path = os.path.join(vis_dir, f"{name}_custom_contours.png")
            cv2.imwrite(contour_path, contour_image)
            
            print(f"[*] Visualization saved: {vis_dir}")
        except Exception as e:
            print(f"[!] Error creating visualization: {str(e)}")
    
    def detect_wall(self, image_path):
        """
        检测图像中的墙面区域
        
        Args:
            image_path: 输入图像路径
            
        Returns:
            墙面区域的二值掩码
        """
        print(f"[*] Detecting wall in: {image_path} with custom algorithm")
        
        # # 检查是否为纯墙面图像
        # filename = os.path.basename(image_path).lower()
        # if 'wall' in filename or 'brick' in filename:
        #     print(f"[*] Detected wall-only image, using direct wall detection")
        #     image = cv2.imread(image_path)
        #     wall_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        #     self._create_visualization(image_path, wall_mask)
        #     return wall_mask
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # 预处理：平滑图像减少噪声
        smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)
        
        # 选择种子点
        seed_points = self._select_seed_points(smoothed_image)
        
        # 开始区域增长算法
        initial_mask = self._region_growing(smoothed_image, seed_points)
        
        # 优化掩码
        refined_mask = self._refine_mask(initial_mask, image)
        
        # 创建可视化
        self._create_visualization(image_path, refined_mask)
        
        print(f"[*] Wall detection completed for: {image_path}")
        return refined_mask


# ===== 模块2: 纹理替换 (Xinhao) =====

class TextureReplacement:
    """负责使用透视变换替换墙面纹理的模块"""
    
    def __init__(self):
        """初始化纹理替换模块"""
        pass
    
    def _find_wall_contours(self, mask):
        """
        查找墙面掩码中的轮廓
        
        Args:
            mask: 强调墙面区域的二值掩码
            
        Returns:
            找到的轮廓
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 只返回有意义的轮廓（过滤小噪点）
        return [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
    
    def _get_perspective_transform(self, contour, texture_size):
        """
        计算给定轮廓的透视变换
        
        Args:
            contour: 表示墙面区域的轮廓
            texture_size: 纹理的大小
            
        Returns:
            变换矩阵
        """
        # 获取轮廓的近似矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(np.intp)  
        
        # 将点排序为：左上、右上、右下、左下
        box = self._sort_points(box)
        
        # 定义纹理中的对应点
        texture_points = np.array([
            [0, 0],
            [texture_size[0], 0],
            [texture_size[0], texture_size[1]],
            [0, texture_size[1]]
        ], dtype=np.float32)
        
        # 计算透视变换
        M = cv2.getPerspectiveTransform(texture_points, box.astype(np.float32))
        return M
    
    def _sort_points(self, points):
        """
        将矩形点按顺时针顺序排序，从左上角开始
        
        Args:
            points: 要排序的点
            
        Returns:
            排序后的点
        """
        # 按y坐标排序（从上到下）
        sorted_by_y = points[np.argsort(points[:, 1])]
        
        # 获取上部和下部点
        top = sorted_by_y[:2]
        bottom = sorted_by_y[2:]
        
        # 按x坐标排序（从左到右）
        top_sorted = top[np.argsort(top[:, 0])]
        bottom_sorted = bottom[np.argsort(bottom[:, 0])]
        
        # 返回顺序：左上、右上、右下、左下
        return np.array([top_sorted[0], top_sorted[1], bottom_sorted[1], bottom_sorted[0]])
    
    def replace_texture(self, original_image, mask, texture_image):
        """
        替换原始图像中的墙面纹理
        
        Args:
            original_image: 原始图像
            mask: 强调墙面区域的二值掩码
            texture_image: 要应用的纹理图像
            
        Returns:
            替换墙面纹理的图像
        """
        print("[*] Replacing wall texture...")
        
        # 确保掩码是二值图像
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # A. 基于纹理瓦片大小调整策略
        # 读取纹理图像
        texture = cv2.imread(texture_image)
        if texture is None:
            raise ValueError(f"Could not read texture image: {texture_image}")
        
        # B. 分析原始墙面纹理特征
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        wall_texture = cv2.bitwise_and(gray_image, mask)
        
        # 检测原始墙面的纹理方向和特征
        # 可以使用梯度方向直方图或傅里叶变换
        # 简化版：仅计算梯度方向
        sobelx = cv2.Sobel(wall_texture, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(wall_texture, cv2.CV_64F, 0, 1, ksize=3)
        gradient_direction = np.arctan2(sobely, sobelx) * 180 / np.pi
        
        # 计算主梯度方向（简化）
        hist, bins = np.histogram(gradient_direction[mask > 0], bins=18, range=(-180, 180))
        primary_angle = bins[np.argmax(hist)]
        
        # C. 根据墙面大小和特征调整纹理大小和方向
        # 查找墙面轮廓
        contours = self._find_wall_contours(mask)
        
        # 创建结果图像副本
        result = original_image.copy()
        
        for contour in contours:
            # 计算轮廓面积和尺寸
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # 根据墙面大小调整纹理比例
            texture_scale = max(1, min(w, h) / 100)  # 简单的启发式缩放
            
            # 根据墙面尺寸调整纹理大小
            texture_h, texture_w = texture.shape[:2]
            optimal_texture_h = int(texture_h * texture_scale)
            optimal_texture_w = int(texture_w * texture_scale)
            
            # 调整纹理大小
            resized_texture = cv2.resize(texture, (optimal_texture_w, optimal_texture_h))
            
            # 如果需要，旋转纹理以匹配墙面方向
            if abs(primary_angle) > 20:  # 只有在有明显方向时才旋转
                rotation_matrix = cv2.getRotationMatrix2D(
                    (optimal_texture_w/2, optimal_texture_h/2), 
                    primary_angle, 
                    1
                )
                resized_texture = cv2.warpAffine(
                    resized_texture, 
                    rotation_matrix, 
                    (optimal_texture_w, optimal_texture_h)
                )
            
            # 为当前轮廓创建独立掩码
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [contour], 0, 255, -1)
            
            # 为透视变换做准备
            adjusted_texture_size = resized_texture.shape[:2][::-1]  # 宽x高
            M = self._get_perspective_transform(contour, adjusted_texture_size)
            
            # 将纹理透视变换到墙面
            h, w = original_image.shape[:2]
            warped_texture = cv2.warpPerspective(resized_texture, M, (w, h))
            
            # 应用变换后的纹理仅到墙面区域
            warped_mask = cv2.warpPerspective(
                np.ones((adjusted_texture_size[1], adjusted_texture_size[0]), dtype=np.uint8) * 255, 
                M, 
                (w, h)
            )
            warped_mask = cv2.bitwise_and(warped_mask, contour_mask)
            
            # 将墙面区域替换为变换后的纹理
            contour_mask_3ch = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)
            result = np.where(contour_mask_3ch > 0, warped_texture, result)
        
        print("[*] Texture replacement completed")
        return result


# ===== 模块3: 光照调整 (Xinhao的部分职责) =====

class LightingAdjustment:
    """负责在替换的纹理上保持真实光照的模块"""
    
    def __init__(self):
        """初始化光照调整模块"""
        pass
    
    def _estimate_lighting(self, original_image, mask):
        """
        估计原始图像中的光照条件
        
        Args:
            original_image: 原始图像
            mask: 强调墙面区域的二值掩码
            
        Returns:
            估计的光照参数
        """
        # 转换为灰度图进行光照分析
        if len(original_image.shape) > 2:
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = original_image.copy()
        
        # 计算墙面区域的亮度统计
        wall_region = cv2.bitwise_and(gray, mask)
        wall_pixels = wall_region[mask > 0]
        
        if len(wall_pixels) == 0:
            return {
                'brightness_mean': 128,
                'brightness_std': 20,
                'gamma': 1.0,
                'light_direction': None
            }
        
        brightness_mean = np.mean(wall_pixels)
        brightness_std = np.std(wall_pixels)
        
        # 估计gamma作为亮度的函数
        # （简单启发式：较暗区域有更高的gamma）
        gamma = 2.2 - brightness_mean / 128
        gamma = max(0.8, min(1.5, gamma))  # 限定在合理范围内
        
        # 尝试估计光源方向
        # 简化版：基于亮度梯度
        sobelx = cv2.Sobel(wall_region, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(wall_region, cv2.CV_64F, 0, 1, ksize=5)
        
        # 使用亮度梯度检测光源方向
        # 亮度增加的方向通常指向光源
        if np.sum(np.abs(sobelx)) > 0 and np.sum(np.abs(sobely)) > 0:
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            # 只考虑梯度较大的区域（更可能表示光照变化而非纹理）
            threshold = np.percentile(gradient_magnitude[mask > 0], 75)
            high_gradient_mask = gradient_magnitude > threshold
            
            # 计算加权平均梯度方向
            weights = gradient_magnitude[high_gradient_mask & (mask > 0)]
            dx = np.mean(sobelx[high_gradient_mask & (mask > 0)] * weights) if weights.size > 0 else 0
            dy = np.mean(sobely[high_gradient_mask & (mask > 0)] * weights) if weights.size > 0 else 0
            
            if dx != 0 or dy != 0:
                light_direction = np.arctan2(dy, dx) * 180 / np.pi
            else:
                light_direction = None
        else:
            light_direction = None
        
        return {
            'brightness_mean': brightness_mean,
            'brightness_std': brightness_std,
            'gamma': gamma,
            'light_direction': light_direction
        }
    
    def _apply_ambient_occlusion(self, image, mask):
        """
        应用简单的环境光遮蔽模拟，增强边缘和角落阴影
        
        Args:
            image: 输入图像
            mask: 墙面掩码
            
        Returns:
            应用了环境光遮蔽的图像
        """
        # 创建掩码边缘的膨胀和腐蚀版本
        kernel = np.ones((15, 15), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        
        # 边缘区域是膨胀和腐蚀之间的差异
        edges = cv2.bitwise_xor(dilated, mask)
        corners = cv2.bitwise_xor(mask, eroded)
        
        # 创建结果图像副本
        result = image.copy().astype(np.float32)
        
        # 在边缘应用轻微暗化
        edge_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) / 255.0
        result = result * (1.0 - edge_mask * 0.2)
        
        # 在角落应用更强的暗化
        corner_mask = cv2.cvtColor(corners, cv2.COLOR_GRAY2BGR) / 255.0
        result = result * (1.0 - corner_mask * 0.3)
        
        return result.astype(np.uint8)
    
    def _apply_directional_lighting(self, image, mask, light_direction):
        """
        应用方向性光照效果
        
        Args:
            image: 输入图像
            mask: 墙面掩码
            light_direction: 光源方向（角度，0表示从右侧照射）
            
        Returns:
            应用了方向性光照的图像
        """
        if light_direction is None:
            return image
        
        # 创建线性梯度
        h, w = mask.shape[:2]
        y, x = np.mgrid[0:h, 0:w]
        
        # 将光源方向转换为弧度
        rad = light_direction * np.pi / 180.0
        
        # 创建方向梯度 (垂直于光源方向)
        gradient = x * np.cos(rad) + y * np.sin(rad)
        
        # 归一化梯度
        gradient = gradient - gradient.min()
        max_grad = gradient.max()
        if max_grad > 0:
            gradient = gradient / max_grad
        
        # 应用方向性光照
        gradient = gradient.reshape(h, w, 1) * 0.4 + 0.8  # 限制光照变化范围
        
        # 只在掩码区域应用光照效果
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = image.copy().astype(np.float32)
        result = result * (gradient * mask_3ch + (1 - mask_3ch))
        
        return result.astype(np.uint8)
    
    def adjust_lighting(self, replaced_image, original_image, mask):
        """
        调整替换纹理的光照以匹配原始图像
        
        Args:
            replaced_image: 替换了纹理的图像
            original_image: 原始图像
            mask: 强调墙面区域的二值掩码
            
        Returns:
            调整了光照的图像
        """
        print("[*] Adjusting lighting...")
        
        # 估计光照条件
        lighting_params = self._estimate_lighting(original_image, mask)
        
        # 创建结果图像副本
        result = replaced_image.copy()
        
        # 应用gamma校正
        # 转换为float32处理
        replaced_float = result.astype(np.float32) / 255.0
        
        # 应用gamma校正
        gamma_corrected = np.power(replaced_float, lighting_params['gamma'])
        
        # 转换回uint8
        gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
        
        # 使用掩码与原始图像混合
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) < 3 else mask
        
        # 边缘的alpha混合以实现平滑过渡
        kernel = np.ones((21, 21), np.uint8)
        edge_mask = cv2.dilate(mask, kernel) - mask
        edge_mask_3ch = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR) if len(edge_mask.shape) < 3 else edge_mask
        
        # 边缘与原始图像混合
        alpha = 0.7  # 混合因子
        blended_edges = cv2.addWeighted(
            gamma_corrected, alpha,
            original_image, 1 - alpha,
            0, dtype=cv2.CV_8U
        )
        
        # 应用混合边缘
        final_result = np.where(edge_mask_3ch > 0, blended_edges, gamma_corrected)
        
        # 对于非墙面区域使用原始图像
        final_result = np.where(mask_3ch > 0, final_result, original_image)
        
        # 应用环境光遮蔽效果
        final_result = self._apply_ambient_occlusion(final_result, mask)
        
        # 应用方向性光照效果
        if lighting_params['light_direction'] is not None:
            final_result = self._apply_directional_lighting(
                final_result, 
                mask, 
                lighting_params['light_direction']
            )
        
        print("[*] Lighting adjustment completed")
        return final_result


# ===== 主系统集成 (Josh) =====

class RepaintSystem:
    """集成所有模块的主系统"""
    
    def __init__(self, color_tolerance=15, shadow_tolerance=30, min_region_size=1000):
        """初始化Repaint系统"""
        self.wall_detector = CustomWallDetector(
            color_tolerance=color_tolerance,
            shadow_tolerance=shadow_tolerance,
            min_region_size=min_region_size
        )
        self.replacement = TextureReplacement()
        self.lighting = LightingAdjustment()
    
    def process_image(self, image_path, texture_path, output_path=None):
        """
        处理图像以替换墙面纹理
        
        Args:
            image_path: 输入图像路径
            texture_path: 纹理图像路径
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
        
        # 1. 检测墙面区域
        mask = self.wall_detector.detect_wall(image_path)
        
        # 2. 读取原始图像
        original_image = cv2.imread(image_path)
        
        # 3. 替换纹理
        replaced_image = self.replacement.replace_texture(original_image, mask, texture_path)
        
        # 4. 调整光照
        final_image = self.lighting.adjust_lighting(replaced_image, original_image, mask)
        
        # 5. 保存结果
        cv2.imwrite(output_path, final_image)
        
        # 6. 创建对比可视化
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
    
    def batch_process(self, image_dir, texture_path, output_dir=None):
        """
        批量处理目录中的所有图像
        
        Args:
            image_dir: 包含输入图像的目录
            texture_path: 纹理图像路径
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
            
            self.process_image(image_path, texture_path, output_path)
        
        print(f"\n=== 批量处理完成。结果已保存至: {output_dir} ===")


# ===== 命令行界面 =====

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RepaintCustom: 墙面识别与替换系统（自定义算法版）')
    
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
    
    # 检查输入图像是否存在
    if not os.path.exists(args.image):
        print(f"错误: 未找到输入图像/目录: {args.image}")
        return
    
    # 检查纹理是否存在
    if not os.path.exists(args.texture):
        print(f"错误: 未找到纹理图像: {args.texture}")
        return
    
    # 处理图像
    if args.batch:
        if not os.path.isdir(args.image):
            print(f"错误: 使用--batch时，--image必须是目录")
            return
        
        repaint.batch_process(args.image, args.texture, args.output)
    else:
        repaint.process_image(args.image, args.texture, args.output)
    
    print("\n=== 处理成功完成 ===")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\n总运行时间: {elapsed_time:.2f} 秒")
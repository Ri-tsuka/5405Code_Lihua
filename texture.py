"""
RepaintCustom: 墙面识别与替换系统（基于用户初选+算法细化）- 第二部分

此部分包含纹理替换模块和光照调整模块。
"""

import os
import sys
import numpy as np
import cv2
import time

# ===== 纹理替换模块 =====

class TextureReplacement:
    """负责使用透视变换替换墙面纹理的模块"""
    
    def __init__(self):
        """初始化纹理替换模块"""
        pass
    
    def _find_wall_contours(self, mask):
        """查找墙面掩码中的轮廓"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
    
    def _get_perspective_transform(self, contour, texture_size):
        """计算给定轮廓的透视变换"""
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(np.intp)  
        
        box = self._sort_points(box)
        
        texture_points = np.array([
            [0, 0],
            [texture_size[0], 0],
            [texture_size[0], texture_size[1]],
            [0, texture_size[1]]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(texture_points, box.astype(np.float32))
        return M
    
    def _sort_points(self, points):
        """将矩形点按顺时针顺序排序，从左上角开始"""
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
        print("[*] 替换墙面纹理...")
        
        # 确保掩码是二值图像
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # 读取纹理图像
        texture = cv2.imread(texture_image)
        if texture is None:
            raise ValueError(f"无法读取纹理图像: {texture_image}")
        
        # 分析原始墙面特征
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        wall_texture = cv2.bitwise_and(gray_image, mask)
        
        # 检测原始墙面的纹理方向和特征
        sobelx = cv2.Sobel(wall_texture, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(wall_texture, cv2.CV_64F, 0, 1, ksize=3)
        gradient_direction = np.arctan2(sobely, sobelx) * 180 / np.pi
        
        # 计算主梯度方向
        hist, bins = np.histogram(gradient_direction[mask > 0], bins=18, range=(-180, 180))
        primary_angle = bins[np.argmax(hist)]
        
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
        
        print("[*] 纹理替换完成")
        return result


# ===== 光照调整模块 =====

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
        gamma = 2.2 - brightness_mean / 128
        gamma = max(0.8, min(1.5, gamma))  # 限定在合理范围内
        
        # 尝试估计光源方向
        sobelx = cv2.Sobel(wall_region, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(wall_region, cv2.CV_64F, 0, 1, ksize=5)
        
        # 使用亮度梯度检测光源方向
        if np.sum(np.abs(sobelx)) > 0 and np.sum(np.abs(sobely)) > 0:
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            threshold = np.percentile(gradient_magnitude[mask > 0], 75)
            high_gradient_mask = gradient_magnitude > threshold
            
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
        print("[*] 调整光照...")
        
        # 估计光照条件
        lighting_params = self._estimate_lighting(original_image, mask)
        
        # 创建结果图像副本
        result = replaced_image.copy()
        
        # 应用gamma校正
        replaced_float = result.astype(np.float32) / 255.0
        gamma_corrected = np.power(replaced_float, lighting_params['gamma'])
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
        
        print("[*] 光照调整完成")
        return final_result
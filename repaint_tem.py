import sys
import numpy as np
import cv2
# ---- GUI环境：PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QGroupBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

# ------------------- Gaussian Blur ------------------------------------------------- 手动实现的点（1）

def generate_gaussian_kernel(ksize=5, sigma=1.0):
    
    # 生成一个二维高斯核，用于模糊图像。
    # 参数：
    #     ksize: 卷积核大小（必须是奇数，如 3, 5, 7）
    #     sigma: 高斯分布的标准差（控制模糊强度，越大越模糊）
    # 返回：
    #     归一化后的高斯核（二维数组）
    

    # 创建一个 1D 坐标数组：从 -ksize//2 到 +ksize//2
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    
    # 创建一个网格坐标（xx, yy），用于二维高斯函数
    xx, yy = np.meshgrid(ax, ax)

    # 根据二维高斯公式计算每个位置的值
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))

    # 将核归一化，使得所有元素加起来为 1（避免改变图像亮度）
    return kernel / np.sum(kernel)

def convolve2d(image, kernel):
    
    # 使用二维卷积操作将图像与给定的 kernel 进行卷积。
    # 参数：
    #     image: 输入灰度图像（二维）
    #     kernel: 卷积核（二维）
    # 返回：
    #     卷积后的图像（与原图大小相同）
    
    h, w = image.shape
    kh, kw = kernel.shape

    # 计算需要填充的边界大小（为了保持尺寸不变）
    pad_h, pad_w = kh // 2, kw // 2

    # 使用 'reflect' 模式填充边缘（镜像边缘以减少边缘效应）
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    # 创建空白输出图像（浮点数更精确）
    result = np.zeros_like(image, dtype=np.float32)

    # 对每个像素进行卷积
    for i in range(h):
        for j in range(w):
            # 取出与卷积核大小相同的局部区域（ROI）
            roi = padded[i:i+kh, j:j+kw]

            # 将局部区域与卷积核相乘并求和，得到新像素值
            result[i, j] = np.sum(roi * kernel)

    # 将结果裁剪到合法的像素值范围 [0, 255]，并转换为整型
    return np.clip(result, 0, 255).astype(np.uint8)


# ------------------- Perspective Warp ------------------------------------------------- 手动实现的点（2）

def warp_perspective_manual(src, M, out_shape):
    
    # 手动实现透视变换（warp），使用逆变换将输出图像映射回源图像。
    # 参数：
    #     src: 原始图像（RGB 或 BGR）
    #     M: 3x3 透视变换矩阵（从原图 -> 新图）
    #     out_shape: 输出图像的尺寸（高度, 宽度）
    # 返回：
    #     result: 应用透视变换后的图像（同通用 API 效果相同）
    
    h_out, w_out = out_shape
    result = np.zeros((h_out, w_out, 3), dtype=np.uint8)

    # 先对变换矩阵求逆（因为我们要从输出图找到原图位置）
    M_inv = np.linalg.inv(M)

    # 遍历输出图的每一个像素
    for y in range(h_out):
        for x in range(w_out):
            # 构造齐次坐标（x, y, 1）
            pt = np.array([x, y, 1.0])

            # 通过逆矩阵映射回原图位置
            mapped = M_inv @ pt

            # 除以第三个元素归一化，得到二维坐标 (u, v)
            mapped /= mapped[2]
            u, v = mapped[:2]

            # 检查是否在原图合法范围内（避免越界）
            if 0 <= u < src.shape[1] - 1 and 0 <= v < src.shape[0] - 1:
                # 找到最近的整数像素索引
                i, j = int(v), int(u)

                # 计算小数部分 a, b（用于双线性插值）
                a, b = v - i, u - j

                # 顶部行插值：两列之间线性插值
                top = (1 - b) * src[i, j] + b * src[i, j + 1]

                # 底部行插值
                bottom = (1 - b) * src[i + 1, j] + b * src[i + 1, j + 1]

                # 两行之间再插值，得到最终像素值
                result[y, x] = ((1 - a) * top + a * bottom).astype(np.uint8)

    return result


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Repaint Demo")

        self.image = cv2.cvtColor(cv2.imread('test2.jpg'), cv2.COLOR_BGR2RGB)
        self.replacement = cv2.cvtColor(cv2.imread('painting.jpg'), cv2.COLOR_BGR2RGB)
        self.wallpaper = cv2.cvtColor(cv2.imread('wallpaper.jpg'), cv2.COLOR_BGR2RGB)
        self.image_copy = self.image.copy()
        self.h_img, self.w_img = self.image.shape[:2]

        self.clicked_points = []
        self.flood_seed = None
        self.target_rgb = np.array([30, 80, 120])
        self.mode = None

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.mousePressEvent = self.on_click

        self.init_ui()
        self.update_display()

    def init_ui(self):
        self.paint_btn = QPushButton("Replace Painting")
        self.colour_btn = QPushButton("Recolour Wall")
        self.wall_btn = QPushButton("Change Wallpaper")
        self.apply_btn = QPushButton("APPLY")
        self.reset_btn = QPushButton("RESET")

        self.paint_btn.clicked.connect(self.enter_paint_mode)
        self.colour_btn.clicked.connect(self.enter_colour_mode)
        self.wall_btn.clicked.connect(self.enter_wall_mode)
        self.apply_btn.clicked.connect(self.apply_current_mode)
        self.reset_btn.clicked.connect(self.reset)

        colours = [([120, 30, 30], 'Wine Red'), ([30, 80, 120], 'Gray Blue'), ([40, 120, 80], 'Gray Green')]
        self.colour_buttons = []
        colour_layout = QVBoxLayout()
        for rgb, label in colours:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, c=rgb: self.set_colour(c))
            colour_layout.addWidget(btn)
            self.colour_buttons.append(btn)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.paint_btn)
        button_layout.addWidget(self.colour_btn)
        button_layout.addWidget(self.wall_btn)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()

        control_box = QGroupBox()
        control_layout = QVBoxLayout()
        control_layout.addLayout(button_layout)
        control_layout.addSpacing(20)
        control_layout.addLayout(colour_layout)
        control_box.setLayout(control_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.label, stretch=4)
        main_layout.addWidget(control_box, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def set_colour(self, rgb):
        self.target_rgb = np.array(rgb)
        print(f"🎨 Target colour set to: {rgb}")

    def update_display(self):
        qimage = QImage(self.image_copy.data, self.image_copy.shape[1], self.image_copy.shape[0],
                        self.image_copy.strides[0], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimage))

    def on_click(self, event):
        x = int(event.pos().x() * self.w_img / self.label.width())
        y = int(event.pos().y() * self.h_img / self.label.height())
        if self.mode in ['paint', 'wallpaper']:
            if len(self.clicked_points) < 4:
                self.clicked_points.append([x, y])
                print(f"Point {len(self.clicked_points)} selected: ({x}, {y})")
            elif self.mode == 'wallpaper' and self.flood_seed is None:
                self.flood_seed = (x, y)
                print(f"FloodFill seed point set to: {self.flood_seed}")
        elif self.mode == 'colour':
            self.flood_seed = (x, y)
            print(f"Recolour seed point set to: {self.flood_seed}")
        self.update_display()

    def enter_paint_mode(self):
        self.mode = 'paint'
        self.clicked_points = []
        self.flood_seed = None
        print("Mode: Replace Painting. Please click 4 corners (clockwise) of the painting.")

    def enter_colour_mode(self):
        self.mode = 'colour'
        self.clicked_points = []
        self.flood_seed = None
        print("Mode: Recolour Wall. Please click the wall area.")

    def enter_wall_mode(self):
        self.mode = 'wallpaper'
        self.clicked_points = []
        self.flood_seed = None
        print("Mode: Wallpaper. Please click 4 corners (clockwise) and a seed point inside the area.")

    def apply_current_mode(self):
        print(f"Applying mode: {self.mode}...")
        if self.mode == 'paint':
            self.apply_painting()
        elif self.mode == 'colour':
            self.apply_colour()
        elif self.mode == 'wallpaper':
            self.apply_wallpaper()

    # --------------------- 将绘画（图片）替换到墙面四边形区域 ---------------------------
    def apply_painting(self):
        # 确保用户已经点击了 4 个角点（顺时针）
        if len(self.clicked_points) != 4:
            print("Please select 4 corners (clockwise) for painting replacement.")
            return

        # 将点击点转为 numpy 数组（目标图像中的 4 个角点）
        dst_pts = np.array(self.clicked_points, dtype=np.float32)

        # 源图（替换用的画）对应的 4 个角点
        src_pts = np.array([
            [0, 0], 
            [self.replacement.shape[1], 0],
            [self.replacement.shape[1], self.replacement.shape[0]],
            [0, self.replacement.shape[0]]
        ], dtype=np.float32)

        # 计算透视变换矩阵 M：从 replacement 映射到 dst_pts 区域
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # 应用手动透视变换，生成贴图后的画作
        warped = warp_perspective_manual(self.replacement, M, (self.h_img, self.w_img))

        # 创建一个用于混合的 mask：表示贴图区域
        mask = np.zeros((self.h_img, self.w_img), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)  # 在区域内填充 255

        # 扩展为三通道，并转换为布尔型 mask
        mask_3ch = np.stack([mask] * 3, axis=-1).astype(bool)

        # 在 mask 区域内替换原图像为 warped 图像
        self.image_copy[mask_3ch] = warped[mask_3ch]

        # 更新 UI 显示
        self.update_display()
        print("✅ Paint replacement applied.")

    #------------------ 刷墙 ----------------------------------
    def apply_colour(self):
        # 确保用户点击了一个种子点
        if self.flood_seed is None:
            print("Please select a seed point for recolouring.")
            return

        # 创建 floodFill 所需的 mask（需比原图大 2 像素）
        mask = np.zeros((self.h_img + 2, self.w_img + 2), np.uint8)

        # 设置 floodFill 的标志（只生成 mask，不改图像）
        flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)

        # 执行 floodFill，在 mask 中填充墙面区域
        cv2.floodFill(self.image_copy.copy(), mask, seedPoint=self.flood_seed, newVal=(0, 0, 0),
                    loDiff=(2,) * 3, upDiff=(2,) * 3, flags=flags)

        # 裁剪掉边框 padding（恢复为原图尺寸）
        mask = mask[1:-1, 1:-1]

        # 对 mask 进行高斯模糊（21x21 核，sigma=5）
        kernel = generate_gaussian_kernel(ksize=21, sigma=5)
        mask_float = convolve2d(mask.astype(np.float32), kernel)

        # 将模糊后的 mask 归一化到 [0, 1] 范围，并扩展为三通道
        mask_blur = np.clip(mask_float / 255.0, 0, 1)[..., np.newaxis]

        # 将当前图像转为灰度（用于保留明暗结构）--------------------------------------- 手动实现的点（3）
        gray = cv2.cvtColor(self.image_copy, cv2.COLOR_RGB2GRAY)
        gray_norm = gray.astype(np.float32) / 255.0
        gray_3ch = np.stack([gray_norm] * 3, axis=-1)  # 转为 3 通道

        # 使用用户选择的颜色 target_rgb 重新上色（调色 = 颜色 × 明度）
        coloured = (self.target_rgb * gray_3ch).astype(np.float32)

        # 原图像转换为 float
        img_float = self.image_copy.astype(np.float32)

        # 混合：模糊区域使用新颜色，其他地方保留原图
        self.image_copy = (mask_blur * coloured + (1 - mask_blur) * img_float).astype(np.uint8)

        # 更新显示
        self.update_display()
        print("✅ Recolouring complete.")

    #----------------- 壁纸 -------------------------------
    def apply_wallpaper(self):
        # 需要点击 4 个角点 + 1 个种子点
        if len(self.clicked_points) != 4 or self.flood_seed is None:
            print("Please select 4 corners and a seed point.")
            return

        # 多边形目标区域（点击点）
        dst_pts = np.array(self.clicked_points, dtype=np.float32)

        # 原图角点，用于计算透视变换（等同于整个图）
        src_pts = np.array([
            [0, 0], 
            [self.w_img, 0], 
            [self.w_img, self.h_img], 
            [0, self.h_img]
        ], dtype=np.float32)

        # 获取透视矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # 将墙纸图片 resize 为和原图一样大
        wallpaper_resized = cv2.resize(self.wallpaper, (self.w_img, self.h_img))

        # 透视变换墙纸
        warped = warp_perspective_manual(wallpaper_resized, M, (self.h_img, self.w_img))

        # 创建多边形 mask（多边形内部区域）
        poly_mask = np.zeros((self.h_img, self.w_img), dtype=np.uint8)
        cv2.fillConvexPoly(poly_mask, dst_pts.astype(np.int32), 255)

        # 创建 floodfill mask（需要加 padding）
        flood_mask = np.zeros((self.h_img + 2, self.w_img + 2), np.uint8)
        cv2.floodFill(self.image.copy(), flood_mask, seedPoint=self.flood_seed,
                    newVal=(0, 0, 0), loDiff=(2,) * 3, upDiff=(2,) * 3,
                    flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))

        # 裁剪 flood mask，恢复原大小
        flood_mask = flood_mask[1:-1, 1:-1]

        # 将 flood 区域与多边形区域求交集，作为最终贴图区域
        final_mask = np.logical_and(poly_mask > 0, flood_mask > 0).astype(np.float32)

        # 模糊该 mask 以实现自然过渡
        kernel = generate_gaussian_kernel(ksize=21, sigma=5)
        mask_float = convolve2d(final_mask * 255, kernel)
        mask_blur = np.clip(mask_float / 255.0, 0, 1)[..., np.newaxis]

        # 将图像转为 float 并执行混合
        image_float = self.image_copy.astype(np.float32)
        warped_float = warped.astype(np.float32)

        # 混合贴图区域与原图（防止边缘生硬）
        blended = (mask_blur * warped_float + (1 - mask_blur) * image_float).astype(np.uint8)

        # 更新图像并刷新显示
        self.image_copy = blended
        self.update_display()
        print("✅ Wallpaper applied.")

    def reset(self):
        self.image_copy = self.image.copy()
        self.clicked_points = []
        self.flood_seed = None
        self.mode = None
        self.update_display()
        print("Reset complete.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

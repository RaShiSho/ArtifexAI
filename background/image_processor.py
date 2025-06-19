import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
import torch.nn.functional as F


class ImageProcessor:
    def __init__(self):
        """
        图像处理器，负责图像合成、颜色匹配和边缘优化
        """
        pass

    def composite_image(self, foreground, background, mask, enhance_edges=True,
                        feather_strength=3, color_match=True):
        """
        合成前景和背景图像

        Args:
            foreground: 前景图像 (BGR)
            background: 背景图像 (BGR)
            mask: 前景掩码 (0-255)
            enhance_edges: 是否增强边缘
            feather_strength: 羽化强度 (1-5)
            color_match: 是否进行颜色匹配

        Returns:
            composite: 合成后的图像
        """
        # 确保图像尺寸一致
        h, w = foreground.shape[:2]
        if background.shape[:2] != (h, w):
            background = cv2.resize(background, (w, h))

        # 预处理掩码
        mask_processed = self._process_mask(mask, feather_strength)

        # 边缘增强
        if enhance_edges:
            mask_processed = self._enhance_mask_edges(foreground, mask_processed)

        # 颜色匹配
        if color_match:
            foreground = self._match_colors(foreground, background, mask_processed)

        # 执行合成
        composite = self._blend_images(foreground, background, mask_processed)

        return composite

    def _process_mask(self, mask, feather_strength):
        """
        处理掩码：羽化、平滑边缘
        """
        # 将掩码转换为浮点数
        mask_float = mask.astype(np.float32) / 255.0

        # 羽化处理
        if feather_strength > 0:
            # 计算羽化核大小
            kernel_size = feather_strength * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            # 距离变换用于羽化
            mask_binary = (mask > 127).astype(np.uint8)
            distance_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)

            # 创建羽化效果
            max_distance = feather_strength * 2
            feathered = np.clip(distance_transform / max_distance, 0, 1)

            # 反向距离变换用于外部羽化
            mask_inv = 1 - mask_binary
            distance_transform_inv = cv2.distanceTransform(mask_inv, cv2.DIST_L2, 5)
            feathered_inv = 1 - np.clip(distance_transform_inv / max_distance, 0, 1)

            # 组合内外羽化
            mask_float = np.minimum(feathered, feathered_inv)

        # 高斯模糊进一步平滑
        mask_float = cv2.GaussianBlur(mask_float, (5, 5), 1.0)

        return mask_float

    def _enhance_mask_edges(self, image, mask):
        """
        基于图像边缘信息增强掩码边缘
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 计算梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)

        # 在掩码边缘区域应用梯度信息
        # 找到掩码边缘
        mask_edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        mask_edges = cv2.dilate(mask_edges, np.ones((5, 5), np.uint8), iterations=1)
        edge_region = mask_edges > 0

        # 在边缘区域根据梯度调整掩码
        enhanced_mask = mask.copy()
        gradient_threshold = 0.3

        edge_pixels = np.where(edge_region)
        for y, x in zip(edge_pixels[0], edge_pixels[1]):
            if gradient_magnitude[y, x] > gradient_threshold:
                # 强边缘，保持掩码值
                continue
            else:
                # 弱边缘，降低掩码值以实现更好的混合
                enhanced_mask[y, x] *= 0.8

        return enhanced_mask

    def _match_colors(self, foreground, background, mask):
        """
        颜色匹配：调整前景颜色以匹配背景
        """
        # 计算前景和背景的平均颜色
        mask_binary = mask > 0.5

        if not np.any(mask_binary):
            return foreground

        # 前景平均颜色
        fg_mean = np.mean(foreground[mask_binary], axis=0)

        # 背景平均颜色
        bg_mask = ~mask_binary
        if np.any(bg_mask):
            bg_mean = np.mean(background[bg_mask], axis=0)
        else:
            bg_mean = np.mean(background, axis=(0, 1))

        # 计算颜色差异
        color_diff = bg_mean - fg_mean

        # 应用颜色调整（轻微调整）
        adjustment_strength = 0.3
        color_adjustment = color_diff * adjustment_strength

        # 应用调整
        adjusted_foreground = foreground.astype(np.float32)
        for i in range(3):  # BGR通道
            adjusted_foreground[:, :, i] += color_adjustment[i]

        adjusted_foreground = np.clip(adjusted_foreground, 0, 255).astype(np.uint8)

        return adjusted_foreground

    def _blend_images(self, foreground, background, mask):
        """
        混合前景和背景图像
        """
        # 确保掩码是三通道
        if len(mask.shape) == 2:
            mask = np.stack([mask, mask, mask], axis=2)

        # 转换为浮点数
        fg_float = foreground.astype(np.float32)
        bg_float = background.astype(np.float32)
        mask_float = mask.astype(np.float32)

        # 线性混合
        composite = fg_float * mask_float + bg_float * (1 - mask_float)

        return composite.astype(np.uint8)

    def adjust_lighting(self, image, brightness=0, contrast=1.0, saturation=1.0):
        """
        调整图像的亮度、对比度和饱和度
        """
        # 转换为浮点数
        img_float = image.astype(np.float32)

        # 亮度调整
        img_float += brightness

        # 对比度调整
        img_float = (img_float - 127.5) * contrast + 127.5

        # 饱和度调整
        if saturation != 1.0:
            hsv = cv2.cvtColor(np.clip(img_float, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= saturation
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

        return np.clip(img_float, 0, 255).astype(np.uint8)

    def add_shadow(self, composite, mask, shadow_offset=(10, 10), shadow_blur=15, shadow_opacity=0.3):
        """
        添加阴影效果
        """
        h, w = composite.shape[:2]
        shadow_mask = np.zeros((h, w), dtype=np.float32)

        # 创建阴影掩码
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.0
        else:
            mask_gray = mask

        # 应用偏移
        offset_x, offset_y = shadow_offset
        if offset_x > 0:
            shadow_mask[offset_y:, offset_x:] = mask_gray[:-offset_y, :-offset_x]
        elif offset_x < 0:
            shadow_mask[offset_y:, :offset_x] = mask_gray[:-offset_y, -offset_x:]
        else:
            shadow_mask[offset_y:, :] = mask_gray[:-offset_y, :]

        # 模糊阴影
        shadow_mask = cv2.GaussianBlur(shadow_mask, (shadow_blur, shadow_blur), 0)

        # 应用阴影
        result = composite.copy().astype(np.float32)
        shadow_effect = np.stack([shadow_mask, shadow_mask, shadow_mask], axis=2)
        result = result * (1 - shadow_effect * shadow_opacity)

        return np.clip(result, 0, 255).astype(np.uint8)
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import requests
import os
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn.functional as F


class SegmentationModel:
    def __init__(self, device='cuda'):
        """
        初始化分割模型
        使用DeepLabV3作为主要分割模型，结合边缘检测优化
        """
        self.device = device
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self._load_model()

    def _load_model(self):
        """加载预训练的分割模型"""
        print("Loading segmentation model...")
        try:
            # 使用DeepLabV3预训练模型
            self.model = deeplabv3_resnet50(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            print("DeepLabV3 model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            # 备用方案：使用简单的GrabCut算法
            self.model = None

    def segment_main_object(self, image, use_edge_refinement=True):
        """
        分割图像中的主要对象

        Args:
            image: BGR格式的输入图像
            use_edge_refinement: 是否使用边缘细化

        Returns:
            mask: 二值掩码，前景为255，背景为0
        """
        if self.model is not None:
            return self._segment_with_deeplab(image, use_edge_refinement)
        else:
            return self._segment_with_grabcut(image)

    def _segment_with_deeplab(self, image, use_edge_refinement=True):
        """使用DeepLabV3进行分割"""
        h, w = image.shape[:2]

        # 预处理
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)['out']
            prediction = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

        # 找到最可能的前景类别（通常是人物）
        person_class = 15  # COCO数据集中人物的类别ID
        mask = (prediction == person_class).astype(np.uint8) * 255

        # 如果没有检测到人物，尝试其他常见前景类别
        if np.sum(mask) < h * w * 0.01:  # 如果掩码太小
            # 尝试其他可能的前景类别
            foreground_classes = [15, 16, 17, 18, 19, 20]  # 人、动物等
            combined_mask = np.zeros_like(prediction, dtype=np.uint8)
            for cls in foreground_classes:
                combined_mask = np.logical_or(combined_mask, prediction == cls)
            mask = combined_mask.astype(np.uint8) * 255

        # 形态学操作清理掩码
        mask = self._clean_mask(mask)

        # 边缘细化
        if use_edge_refinement:
            mask = self._refine_edges(image, mask)

        return mask

    def _segment_with_grabcut(self, image):
        """使用GrabCut算法进行分割（备用方案）"""
        h, w = image.shape[:2]

        # 创建初始矩形（假设主体在图像中央）
        margin = min(h, w) // 8
        rect = (margin, margin, w - 2 * margin, h - 2 * margin)

        # 初始化掩码
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # 应用GrabCut
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # 创建最终掩码
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result_mask = mask2 * 255

        return self._clean_mask(result_mask)

    def _clean_mask(self, mask):
        """清理掩码：去噪声、填充洞洞"""
        # 形态学开运算去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 形态学闭运算填充洞洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 找到最大连通区域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 保留最大的连通区域
            largest_contour = max(contours, key=cv2.contourArea)
            mask_clean = np.zeros_like(mask)
            cv2.fillPoly(mask_clean, [largest_contour], 255)
            return mask_clean

        return mask

    def _refine_edges(self, image, mask):
        """使用边缘信息细化掩码边缘"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 计算梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # 标准化梯度
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 在掩码边界附近使用梯度信息
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_dilated = cv2.dilate(mask, kernel, iterations=2)
        mask_eroded = cv2.erode(mask, kernel, iterations=2)
        edge_region = mask_dilated - mask_eroded

        # 在边缘区域应用梯度阈值
        edge_pixels = np.where(edge_region > 0)
        for y, x in zip(edge_pixels[0], edge_pixels[1]):
            if gradient_magnitude[y, x] > 50:  # 强边缘
                # 保持原掩码值
                continue
            else:
                # 弱边缘，可能需要调整
                if mask[y, x] == 255:
                    # 检查周围像素
                    neighborhood = mask[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
                    if np.mean(neighborhood) < 200:  # 周围大部分是背景
                        mask[y, x] = 0

        return mask

    def create_trimap(self, mask, erode_size=10, dilate_size=10):
        """
        创建三元图用于更精确的抠图

        Args:
            mask: 二值掩码
            erode_size: 腐蚀核大小
            dilate_size: 膨胀核大小

        Returns:
            trimap: 三元图 (0=背景, 128=未知, 255=前景)
        """
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))

        # 确定前景区域
        fg_mask = cv2.erode(mask, kernel_erode, iterations=1)

        # 确定背景区域
        bg_mask = cv2.erode(255 - mask, kernel_erode, iterations=1)
        bg_mask = 255 - bg_mask

        # 创建三元图
        trimap = np.zeros_like(mask)
        trimap[fg_mask == 255] = 255  # 确定前景
        trimap[bg_mask == 0] = 0  # 确定背景
        trimap[(fg_mask != 255) & (bg_mask != 0)] = 128  # 未知区域

        return trimap
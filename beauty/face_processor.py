import cv2
import numpy as np
import torch
from gfpgan import GFPGANer
from .skin_enhancement import SkinEnhancer
from .face_detector import FaceDetector

class FaceProcessor:
    """人脸处理类，集成多种人脸增强功能"""
    def __init__(self, model_path='GFPGANv1.4.pth', upscale=2):
        """
                初始化人脸处理器

                参数:
                    model_path: GFPGAN模型路径
                    upscale: 超分辨率放大倍数
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化GFPGAN
        self.gfpgan = GFPGANer(
            model_path=model_path,
            upscale=upscale,   # 超分辨率放大倍数
            arch='clean',    # 使用clean架构
            channel_multiplier=2,  # 通道倍增器
            bg_upsampler=None  # 不使用背景上采样器
        )

        # 初始化其他模块
        self.skin_enhancer = SkinEnhancer()
        self.face_detector = FaceDetector()

    def process_face(self, image_path, output_path=None):
        """
                处理单张含有人脸的图片

                参数:
                    image_path: 输入图片路径
                    output_path: 输出图片路径（可选）

                返回:
                    处理后的图像(numpy数组)
        """
        # 读取图片
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # 检测人脸位置（返回扩展后的人脸框列表）
        faces = self.face_detector.detect_faces(img)

        if len(faces) == 0:
            print("未检测到人脸")
            return img

        # 对每个人脸进行处理
        for face_box in faces:
            x, y, w, h = face_box
            face_roi = img[y:y + h, x:x + w]

            # ----------------------------
            # GFPGAN人脸增强（三步处理）
            # ----------------------------
            # 参数说明:
            # has_aligned: 是否已对齐人脸
            # only_center_face: 是否只处理中心人脸
            _, _, enhanced_face = self.gfpgan.enhance(face_roi, has_aligned=False, only_center_face=False)

            # 肤色均衡
            enhanced_face = self.skin_enhancer.balance_skin_tone(enhanced_face)

            # 磨皮滤波
            enhanced_face = self.skin_enhancer.skin_smoothing(enhanced_face)

            # 将处理后的人脸放回原图
            img[y:y + h, x:x + w] = cv2.resize(enhanced_face, (w, h))

        # 保存结果
        if output_path:
            cv2.imwrite(output_path, img)

        return img
import cv2
import numpy as np
import torch
from gfpgan import GFPGANer
from .skin_enhancement import SkinEnhancer
from .face_detector import FaceDetector



class FaceProcessor:
    def __init__(self, model_path='GFPGANv1.4.pth', upscale=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化GFPGAN
        self.gfpgan = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )

        # 初始化其他模块
        self.skin_enhancer = SkinEnhancer()
        self.face_detector = FaceDetector()

    def process_face(self, image_path, output_path=None):
        """处理单张人脸图片"""
        # 读取图片
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # 检测人脸
        faces = self.face_detector.detect_faces(img)

        if len(faces) == 0:
            print("未检测到人脸")
            return img

        # 对每个人脸进行处理
        for face_box in faces:
            x, y, w, h = face_box
            face_roi = img[y:y + h, x:x + w]

            # GFPGAN美颜
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
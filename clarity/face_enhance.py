
import os
import cv2
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer


class FaceEnhancer:
    def __init__(self, device_id=0):
        # 获取当前文件所在路径
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 构建模型文件的完整路径
        gfpgan_model_path = os.path.join(current_dir, 'models', 'GFPGANv1.4.pth')
        realesrgan_model_path = os.path.join(current_dir, 'models', 'RealESRGAN_x4plus.pth')

        # 初始化 RealESRGAN 背景增强器
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        self.bg_upsampler = RealESRGANer(
            scale=4,
            model_path=realesrgan_model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            gpu_id=device_id
        )

        # 初始化 GFPGAN 人脸增强器
        self.face_enhancer = GFPGANer(
            model_path=gfpgan_model_path,
            upscale=4,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.bg_upsampler
        )

    def enhance(self, input_img: np.ndarray) -> np.ndarray:
        """执行人脸增强，返回增强后图像"""
        if input_img is None or not isinstance(input_img, np.ndarray):
            raise ValueError("Input image is invalid.")

        _, _, output_img = self.face_enhancer.enhance(
            input_img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )
        return output_img

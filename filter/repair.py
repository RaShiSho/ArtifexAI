#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-ESRGAN图像超分辨率处理器
负责图像增强和修复的核心功能
"""

import cv2
import numpy as np
import torch
import os
import logging
from pathlib import Path
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ESRGANProcessor:
    """Real-ESRGAN处理器类"""

    def __init__(self, model_name='RealESRGAN_x4plus', device=None):
        """
        初始化ESRGAN处理器

        Args:
            model_name: 模型名称
            device: 计算设备 ('cpu', 'cuda', 'mps' 或 None 自动选择)
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.model_path = None
        self.is_loaded = False

        # 模型配置
        self.model_urls = {
            'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'RealESRGAN_x2plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            'RealESRGAN_x4plus_anime_6B': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
        }

        # 创建模型目录
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)

        logger.info(f"ESRGAN处理器初始化完成，使用设备: {self.device}")

    def _get_device(self, device=None):
        """获取计算设备"""
        if device:
            return device

        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def download_model(self, model_name=None):
        """下载预训练模型"""
        if model_name is None:
            model_name = self.model_name

        if model_name not in self.model_urls:
            raise ValueError(f"不支持的模型: {model_name}")

        model_path = self.models_dir / f"{model_name}.pth"

        if model_path.exists():
            logger.info(f"模型已存在: {model_path}")
            return str(model_path)

        url = self.model_urls[model_name]
        logger.info(f"正在下载模型: {model_name}")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(model_path, 'wb') as f, tqdm(
                    desc=f"下载 {model_name}",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            logger.info(f"模型下载완成: {model_path}")
            return str(model_path)

        except Exception as e:
            if model_path.exists():
                model_path.unlink()  # 删除不完整的文件
            raise Exception(f"模型下载失败: {str(e)}")

    def load_model(self):
        """加载模型"""
        if self.is_loaded:
            return True

        try:
            # 尝试导入Real-ESRGAN
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                logger.info("使用Real-ESRGAN库")
                use_realesrgan = True
            except ImportError:
                logger.warning("Real-ESRGAN库未安装，使用基础实现")
                use_realesrgan = False

            # 下载模型文件
            self.model_path = self.download_model()

            if use_realesrgan:
                # 使用官方Real-ESRGAN库
                if 'anime' in self.model_name.lower():
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                    num_block=6, num_grow_ch=32, scale=4)
                else:
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                    num_block=23, num_grow_ch=32, scale=4)

                self.model = RealESRGANer(
                    scale=4,
                    model_path=self.model_path,
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=self.device != 'cpu'
                )
            else:
                # 使用基础实现
                self.model = self._load_basic_model()

            self.is_loaded = True
            logger.info("模型加载成功")
            return True

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            return False

    def _load_basic_model(self):
        """加载基础模型实现"""

        # 这里实现一个基础的超分辨率模型
        # 如果没有Real-ESRGAN库，使用OpenCV的超分辨率
        class BasicSRModel:
            def __init__(self, device):
                self.device = device
                # 使用OpenCV的DNN超分辨率
                try:
                    self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
                    # 这里可以加载ESPCN或其他轻量级模型
                    logger.info("使用OpenCV DNN超分辨率")
                except:
                    self.sr = None
                    logger.warning("OpenCV DNN不可用，将使用双三次插值")

            def enhance(self, img, outscale=4):
                if self.sr is not None:
                    try:
                        return self.sr.upsample(img), None
                    except:
                        pass

                # 回退到双三次插值
                h, w = img.shape[:2]
                return cv2.resize(img, (w * outscale, h * outscale),
                                  interpolation=cv2.INTER_CUBIC), None

        return BasicSRModel(self.device)

    def process_image(self, input_path, output_path, scale=4, face_enhance=False):
        """
        处理单张图像

        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径
            scale: 放大倍数
            face_enhance: 是否启用人脸增强

        Returns:
            tuple: (成功标志, 消息)
        """
        try:
            # 检查输入文件
            if not os.path.exists(input_path):
                return False, f"输入文件不存在: {input_path}"

            # 加载模型
            if not self.load_model():
                return False, "模型加载失败"

            # 读取图像
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                return False, "无法读取图像文件"

            logger.info(f"处理图像: {input_path}, 尺寸: {img.shape}")

            # 图像预处理
            img = self._preprocess_image(img)

            # 执行超分辨率
            try:
                if hasattr(self.model, 'enhance'):
                    # 官方Real-ESRGAN
                    output, _ = self.model.enhance(img, outscale=scale)
                else:
                    # 基础实现
                    output, _ = self.model.enhance(img, outscale=scale)

                if output is None:
                    return False, "图像增强失败"

            except Exception as e:
                logger.error(f"图像增强过程出错: {str(e)}")
                return False, f"图像增强失败: {str(e)}"

            # 后处理（如果需要人脸增强）
            if face_enhance:
                output = self._enhance_faces(output)

            # 保存结果
            success = cv2.imwrite(output_path, output)
            if not success:
                return False, "保存输出图像失败"

            logger.info(f"图像处理完成: {output_path}, 尺寸: {output.shape}")
            return True, "处理成功"

        except Exception as e:
            logger.error(f"处理图像时发生错误: {str(e)}")
            return False, f"处理失败: {str(e)}"

    def _preprocess_image(self, img):
        """图像预处理"""
        # 确保图像在有效范围内
        img = np.clip(img, 0, 255).astype(np.uint8)

        # 如果图像过大，先进行适当缩放
        h, w = img.shape[:2]
        max_size = 2048  # 最大边长

        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"图像预缩放到: {new_w}x{new_h}")

        return img

    def _enhance_faces(self, img):
        """人脸增强（可选功能）"""
        try:
            # 加载人脸检测器
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            # 转换为灰度图进行人脸检测
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                logger.info(f"检测到 {len(faces)} 个人脸，进行增强")

                for (x, y, w, h) in faces:
                    # 提取人脸区域
                    face_roi = img[y:y + h, x:x + w]

                    # 对人脸区域进行额外的锐化处理
                    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    face_roi = cv2.filter2D(face_roi, -1, kernel)

                    # 将增强后的人脸放回原图
                    img[y:y + h, x:x + w] = face_roi

            return img

        except Exception as e:
            logger.warning(f"人脸增强失败，跳过: {str(e)}")
            return img

    def get_model_status(self):
        """获取模型状态"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'model_path': self.model_path
        }

    def cleanup(self):
        """清理资源"""
        if self.model is not None:
            del self.model
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_loaded = False
        logger.info("ESRGAN处理器资源已清理")
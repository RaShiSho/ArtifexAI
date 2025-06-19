import torch
import cv2
from .segmentation_model import SegmentationModel
from .enhanced_background_generator import EnhancedBackgroundGenerator
from .image_processor import ImageProcessor

class BackgroundReplacementSystem:
    """集成的背景替换系统"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")

        # 初始化模块，支持模型懒加载
        self.segmentation_model = None
        self.background_generator = None
        self.image_processor = None
        self._models_loaded = False

    def _ensure_models_loaded(self):
        """确保模型已加载，支持懒加载"""
        if not self._models_loaded:
            print("Loading background replacement models...")
            try:
                self.segmentation_model = SegmentationModel(device=self.device)
                self.background_generator = EnhancedBackgroundGenerator(device=self.device)
                self.image_processor = ImageProcessor()
                self._models_loaded = True
                print("Background replacement models loaded successfully!")
            except Exception as e:
                print(f"Error loading background replacement models: {e}")
                raise

    def process_image(self, input_path, output_path, background_style='nature',
                      enhance_edges=True, feather_strength=3, quality='medium'):
        """
        处理单张图片：分割 + 背景替换

        Args:
            input_path: 输入图片路径
            output_path: 输出图片路径
            background_style: 背景风格 ('nature', 'urban', 'abstract', 'fantasy', 'space', 'vintage')
            enhance_edges: 是否增强边缘
            feather_strength: 羽化强度 (1-5)
            quality: 生成质量 ('low', 'medium', 'high')
        """
        try:
            # 确保模型已加载
            self._ensure_models_loaded()

            # 1. 加载图片
            print(f"Loading image: {input_path}")
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"Cannot load image: {input_path}")

            # 2. 对象分割
            print("Performing segmentation...")
            mask = self.segmentation_model.segment_main_object(image)

            # 3. 生成增强背景
            print(f"Generating {background_style} background with {quality} quality...")
            h, w = image.shape[:2]
            background = self.background_generator.generate_background(
                width=w, height=h, style=background_style, quality=quality
            )

            # 4. 图片处理和合成
            print("Processing and compositing...")
            result = self.image_processor.composite_image(
                foreground=image,
                background=background,
                mask=mask,
                enhance_edges=enhance_edges,
                feather_strength=feather_strength
            )

            # 5. 保存结果
            cv2.imwrite(output_path, result)
            print(f"Result saved to: {output_path}")

            return result

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise

    def get_available_styles(self):
        """获取可用的背景风格"""
        # 确保背景生成器已加载
        if not self._models_loaded:
            self._ensure_models_loaded()
        return self.background_generator.get_available_styles()

    def cleanup(self):
        """清理资源"""
        if self.background_generator is not None:
            self.background_generator.cleanup_models()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
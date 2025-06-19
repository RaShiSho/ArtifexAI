import torch
import cv2
from .segmentation_model import SegmentationModel
from .enhanced_background_generator import EnhancedBackgroundGenerator
from .image_processor import ImageProcessor


class BackgroundReplacementSystem:
    def __init__(self, device=None):
        """
        初始化背景替换系统。
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Initializing BackgroundReplacementSystem on device: {self.device}")

        # 初始化功能模块
        self.segmentation_model = SegmentationModel(device=self.device)
        self.background_generator = EnhancedBackgroundGenerator(device=self.device)
        self.image_processor = ImageProcessor()

    def process_image(self, input_path, output_path, background_style='nature',
                      enhance_edges=True, feather_strength=3, quality='medium'):
        try:
            # 1. 加载图片
            print(f"Loading image: {input_path}")
            image = cv2.imread(str(input_path))
            if image is None:
                raise ValueError(f"Cannot load image: {input_path}")

            # 2. 物体分割，获取蒙版
            print("Performing segmentation...")
            mask = self.segmentation_model.segment_main_object(image)

            # 3. 根据风格生成背景
            print(f"Generating '{background_style}' background with '{quality}' quality...")
            h, w = image.shape[:2]
            background = self.background_generator.generate_background(
                width=w, height=h, style=background_style, quality=quality
            )

            # 4. 图像合成处理
            print("Compositing image...")
            result = self.image_processor.composite_image(
                foreground=image,
                background=background,
                mask=mask,
                enhance_edges=enhance_edges,
                feather_strength=feather_strength
            )

            # 5. 保存结果
            cv2.imwrite(str(output_path), result)
            print(f"Result saved to: {output_path}")

            return True

        except Exception as e:
            print(f"Error during image processing: {str(e)}")
            # 可以在这里引入更详细的日志记录
            raise e

    def get_available_styles(self):
        """获取所有可用的背景风格。"""
        return self.background_generator.get_available_styles()

    def cleanup(self):
        """清理和释放模型资源，在应用关闭时调用。"""
        print("Cleaning up background system resources...")
        self.background_generator.cleanup_models()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
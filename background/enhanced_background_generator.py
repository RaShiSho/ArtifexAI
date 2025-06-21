import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import random
import warnings

warnings.filterwarnings("ignore")

# 尝试导入扩散模型相关库
try:
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import logging
    logging.set_verbosity_error() # 设置日志级别为错误
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Error: diffusers not installed. This package requires diffusers to work.")
    print("To install: pip install diffusers transformers accelerate")
    raise ImportError("diffusers package is required")


class EnhancedBackgroundGenerator:
    """增强背景生成器，使用预训练的AI模型生成高质量背景"""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
                初始化增强背景生成器

                参数:
                    device: 使用的计算设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.stable_diffusion_pipe = None

        # 风格提示词库
        self.style_prompts = {
            'nature': [
                "beautiful landscape, mountains, forest, sunset, cinematic lighting, 8k, high detail",
                "serene nature scene, green meadow, blue sky, fluffy clouds, professional photography",
                "tropical paradise, palm trees, beach, ocean waves, golden hour lighting",
                "autumn forest, colorful leaves, misty morning, soft natural light",
                "mountain valley, snow peaks, alpine lake, dramatic sky, landscape photography",
                "flower field, lavender, sunrise, soft bokeh, dreamy atmosphere"
            ],
            'urban': [
                "modern city skyline, skyscrapers, blue hour, neon lights, urban photography",
                "street photography, bokeh lights, night scene, city atmosphere",
                "architectural photography, glass buildings, geometric patterns, modern design",
                "urban landscape, concrete jungle, dramatic lighting, metropolitan",
                "cityscape at dusk, golden hour, urban environment, professional photography",
                "industrial scene, modern architecture, steel and glass, urban aesthetic"
            ],
            'abstract': [
                "abstract art, flowing colors, gradient, modern digital art, 8k resolution",
                "geometric abstract background, colorful patterns, modern art style",
                "fluid abstract painting, vibrant colors, artistic composition",
                "digital abstract art, neon colors, futuristic design, high resolution",
                "watercolor abstract, soft gradients, artistic background, creative design",
                "minimalist abstract, clean lines, modern color palette, geometric shapes"
            ],
            'fantasy': [
                "enchanted forest, glowing trees, magical light, twilight atmosphere, ultra-detailed, fantasy art",
                "floating castle, misty mountains, fantasy world, cinematic lighting, majestic style",
                "dragon flying over mystic land, sunset, fantasy theme, epic environment, 8k resolution",
                "fairy tale scene, colorful magical village, warm soft light, whimsical background",
                "ancient ruins with mystical aura, magical realism, forest background, enchanted sky"
            ],
            'space': [
                "outer space, galaxy with stars and nebulas, deep cosmic background, ultra high resolution",
                "planet with rings in space, starfield, glowing nebula, science fiction scenery",
                "astronaut floating in space, distant stars, cosmic dust, dramatic lighting",
                "moon landscape, distant earth in the sky, photorealistic space environment",
                "space station window view, starscape background, futuristic sci-fi setting"
            ],
            'vintage': [
                "vintage wallpaper, faded colors, floral pattern, 1970s retro style",
                "old film photography background, sepia tones, grainy texture",
                "classic vintage room, antique furniture, warm nostalgic lighting",
                "rustic wooden wall, old-fashioned decor, retro photography style",
                "vintage studio backdrop, soft tones, aged paper background texture"
            ]
        }

        print(f"Using device: {self.device}")

        # 初始化模型
        if DIFFUSERS_AVAILABLE:
            self._load_stable_diffusion()
        else:
            raise RuntimeError("Required packages not available. Please install diffusers.")

    def _load_stable_diffusion(self):
        """加载Stable Diffusion模型"""
        try:
            print("Loading Stable Diffusion model...")
            model_id = "runwayml/stable-diffusion-v1-5"

            self.stable_diffusion_pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.stable_diffusion_pipe = self.stable_diffusion_pipe.to(self.device)

            # 优化内存使用
            if self.device == 'cuda':
                try:
                    self.stable_diffusion_pipe.enable_memory_efficient_attention()
                except:
                    pass

            print("Stable Diffusion model loaded successfully!")

        except Exception as e:
            print(f"Error loading Stable Diffusion: {e}")
            raise RuntimeError("Failed to load Stable Diffusion model")

    def generate_background(self, width, height, style='nature', seed=None, quality='medium'):
        """
                使用Stable Diffusion生成高质量背景图像
                参数:
                    width: 背景宽度
                    height: 背景高度
                    style: 背景风格 ('nature', 'urban'等)
                    seed: 随机种子(用于可重复结果)
                    quality: 质量设置 ('low', 'medium', 'high')
                返回:
                    background: BGR格式的背景图像
        """
        if seed is not None:  # 设置随机种子
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        if self.stable_diffusion_pipe is None:
            raise RuntimeError("Stable Diffusion model not loaded")

        return self._generate_with_stable_diffusion(width, height, style, quality)

    def _generate_with_stable_diffusion(self, width, height, style, quality):
        """使用Stable Diffusion生成背景"""
        try:
            # 随机选择提示词
            prompt = random.choice(self.style_prompts.get(style, self.style_prompts['nature']))

            # 根据质量添加增强词
            quality_enhancers = {
                'low': "simple, clean",
                'medium': "detailed, high quality, professional",
                'high': "highly detailed, masterpiece, 8k, ultra realistic, professional photography"
            }

            enhanced_prompt = f"{prompt}, {quality_enhancers[quality]}"
            negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, watermark, text, signature"

            # 根据质量调整最大尺寸
            max_size = 768 if quality == 'high' else 512

            # 计算合适的生成尺寸(保持宽高比)
            aspect_ratio = width / height
            if width > height:
                gen_width = min(max_size, width)
                gen_height = int(gen_width / aspect_ratio)
            else:
                gen_height = min(max_size, height)
                gen_width = int(gen_height * aspect_ratio)

            # 确保尺寸是8的倍数(Stable Diffusion要求)
            gen_width = (gen_width // 8) * 8
            gen_height = (gen_height // 8) * 8

            print(f"Generating background with prompt: {enhanced_prompt[:100]}...")

            # 生成图像
            with torch.autocast(self.device):   # 自动混合精度
                result = self.stable_diffusion_pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    width=gen_width,
                    height=gen_height,
                    num_inference_steps=20 if quality == 'low' else 30 if quality == 'medium' else 50,
                    guidance_scale=7.5,
                    generator=torch.Generator(device=self.device).manual_seed(random.randint(0, 1000000))
                )

            # 转换为PIL图像
            pil_image = result.images[0]

            # 如果需要，调整到目标尺寸
            if (gen_width, gen_height) != (width, height):
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)

            # 转换为BGR格式
            background = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            print("Background generated successfully!")
            return background

        except Exception as e:
            print(f"Error in Stable Diffusion generation: {e}")
            raise RuntimeError("Failed to generate background with Stable Diffusion")

    def get_available_styles(self):
        """获取可用的背景风格列表"""
        return list(self.style_prompts.keys())

    def cleanup_models(self):
        """清理模型以释放内存"""
        if self.stable_diffusion_pipe is not None:
            del self.stable_diffusion_pipe
            self.stable_diffusion_pipe = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Models cleaned up, memory released")
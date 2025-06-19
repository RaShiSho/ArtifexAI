import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import random
import warnings

warnings.filterwarnings("ignore")

# Try to import diffusion model related libraries
try:
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import logging
    logging.set_verbosity_error()
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Note: diffusers not installed. Will use basic background generation.")
    print("To install: pip install diffusers transformers accelerate")


class EnhancedBackgroundGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Enhanced background generator using pre-trained AI models
        """
        self.device = device
        self.stable_diffusion_pipe = None

        # Style prompt library
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

        # Basic background generator as fallback
        self.basic_generator = BasicBackgroundGenerator()

        print(f"Using device: {self.device}")

        # Initialize models
        if DIFFUSERS_AVAILABLE:
            self._load_stable_diffusion()

    def _load_stable_diffusion(self):
        """Load Stable Diffusion model"""
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

            # Optimize memory usage
            if self.device == 'cuda':
                try:
                    self.stable_diffusion_pipe.enable_memory_efficient_attention()
                except:
                    pass

            print("Stable Diffusion model loaded successfully!")

        except Exception as e:
            print(f"Error loading Stable Diffusion: {e}")
            print("Falling back to basic background generation")
            self.stable_diffusion_pipe = None

    def generate_background(self, width, height, style='nature', seed=None, quality='medium'):
        """
        Generate high-quality background image

        Args:
            width: Background width
            height: Background height
            style: Background style
            seed: Random seed
            quality: Quality setting ('low', 'medium', 'high')

        Returns:
            background: BGR format background image
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Use Stable Diffusion if available
        if self.stable_diffusion_pipe is not None:
            return self._generate_with_stable_diffusion(width, height, style, quality)
        else:
            # Fallback: use basic generator
            return self.basic_generator.generate_background(width, height, style, seed)

    def _generate_with_stable_diffusion(self, width, height, style, quality):
        """Generate background using Stable Diffusion"""
        try:
            # Select prompt
            prompt = random.choice(self.style_prompts.get(style, self.style_prompts['nature']))

            # Add quality enhancers
            quality_enhancers = {
                'low': "simple, clean",
                'medium': "detailed, high quality, professional",
                'high': "highly detailed, masterpiece, 8k, ultra realistic, professional photography"
            }

            enhanced_prompt = f"{prompt}, {quality_enhancers[quality]}"
            negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, watermark, text, signature"

            # Adjust resolution for memory constraints
            max_size = 768 if quality == 'high' else 512

            # Calculate appropriate generation size (maintain aspect ratio)
            aspect_ratio = width / height
            if width > height:
                gen_width = min(max_size, width)
                gen_height = int(gen_width / aspect_ratio)
            else:
                gen_height = min(max_size, height)
                gen_width = int(gen_height * aspect_ratio)

            # Ensure dimensions are multiples of 8 (Stable Diffusion requirement)
            gen_width = (gen_width // 8) * 8
            gen_height = (gen_height // 8) * 8

            print(f"Generating background with prompt: {enhanced_prompt[:100]}...")

            # Generate image
            with torch.autocast(self.device):
                result = self.stable_diffusion_pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    width=gen_width,
                    height=gen_height,
                    num_inference_steps=20 if quality == 'low' else 30 if quality == 'medium' else 50,
                    guidance_scale=7.5,
                    generator=torch.Generator(device=self.device).manual_seed(random.randint(0, 1000000))
                )

            # Convert to OpenCV format
            pil_image = result.images[0]

            # Resize to target size if needed
            if (gen_width, gen_height) != (width, height):
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)

            # Convert to BGR format
            background = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            print("Background generated successfully!")
            return background

        except Exception as e:
            print(f"Error in Stable Diffusion generation: {e}")
            print("Falling back to basic generation...")
            return self.basic_generator.generate_background(width, height, style)

    def get_available_styles(self):
        """Get list of available background styles"""
        return list(self.style_prompts.keys())

    def cleanup_models(self):
        """Clean up models to free memory"""
        if self.stable_diffusion_pipe is not None:
            del self.stable_diffusion_pipe
            self.stable_diffusion_pipe = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Models cleaned up, memory released")


class BasicBackgroundGenerator:
    """Basic background generator (as fallback)"""

    def __init__(self):
        self.style_configs = {
            'nature': {
                'colors': [(34, 139, 34), (107, 142, 35), (46, 125, 50), (76, 175, 80)],
                'patterns': ['gradient', 'cloud', 'texture'],
                'blur_range': (15, 25)
            },
            'urban': {
                'colors': [(70, 70, 70), (105, 105, 105), (128, 128, 128), (169, 169, 169)],
                'patterns': ['geometric', 'gradient', 'bokeh'],
                'blur_range': (20, 35)
            },
            'portrait': {
                'colors': [(139, 69, 19), (160, 82, 45), (205, 133, 63), (222, 184, 135)],
                'patterns': ['gradient', 'soft_light'],
                'blur_range': (25, 40)
            },
            'abstract': {
                'colors': [(255, 20, 147), (138, 43, 226), (30, 144, 255), (255, 165, 0)],
                'patterns': ['swirl', 'gradient', 'geometric'],
                'blur_range': (10, 30)
            },
            'fantasy': {
                'colors': [(186, 85, 211), (72, 61, 139), (255, 182, 193), (70, 130, 180)],
                'patterns': ['cloud', 'gradient', 'swirl'],
                'blur_range': (20, 35)
            },
            'space': {
                'colors': [(0, 0, 0), (25, 25, 112), (75, 0, 130), (123, 104, 238)],
                'patterns': ['gradient', 'geometric', 'swirl'],
                'blur_range': (15, 30)
            },
            # 'vintage': {
            #     'colors': [(139, 69, 19), (222, 184, 135), (205, 133, 63), (210, 180, 140)],
            #     'patterns': ['gradient', 'texture', 'cloud'],
            #     'blur_range': (10, 25)
            # }

        }

    def generate_background(self, width, height, style='nature', seed=None):
        """Generate basic background"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        config = self.style_configs.get(style, self.style_configs['nature'])
        pattern = random.choice(config['patterns'])

        if pattern == 'gradient':
            return self._generate_gradient(width, height, config['colors'])
        elif pattern == 'cloud':
            return self._generate_cloud(width, height, config['colors'])
        elif pattern == 'geometric':
            return self._generate_geometric(width, height, config['colors'])
        elif pattern == 'bokeh':
            return self._generate_bokeh(width, height, config['colors'])
        else:
            return self._generate_gradient(width, height, config['colors'])

    def _generate_gradient(self, width, height, colors):
        """Generate gradient background"""
        color1 = random.choice(colors)
        color2 = random.choice(colors)

        background = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            ratio = y / height
            for c in range(3):
                background[y, :, c] = int(color1[c] * (1 - ratio) + color2[c] * ratio)

        return background

    def _generate_cloud(self, width, height, colors):
        """Generate cloud texture"""
        base_color = random.choice(colors)
        background = np.full((height, width, 3), base_color, dtype=np.uint8)

        # Add noise
        noise = np.random.randint(-30, 30, (height, width, 3))
        background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Blur
        background = cv2.GaussianBlur(background, (21, 21), 0)

        return background

    def _generate_geometric(self, width, height, colors):
        """Generate geometric patterns"""
        background = np.full((height, width, 3), random.choice(colors), dtype=np.uint8)

        # Add geometric shapes
        for _ in range(random.randint(3, 8)):
            color = random.choice(colors)
            center = (random.randint(0, width), random.randint(0, height))
            radius = random.randint(20, min(width, height) // 4)
            cv2.circle(background, center, radius, color, -1)

        # Blur
        background = cv2.GaussianBlur(background, (25, 25), 0)

        return background

    def _generate_bokeh(self, width, height, colors):
        """Generate bokeh effect"""
        background = np.full((height, width, 3), random.choice(colors), dtype=np.uint8)

        # Add light spots
        for _ in range(random.randint(15, 30)):
            center = (random.randint(0, width), random.randint(0, height))
            radius = random.randint(10, 60)
            color = random.choice(colors)

            overlay = np.zeros_like(background, dtype=np.float32)
            cv2.circle(overlay, center, radius, color, -1)
            overlay = cv2.GaussianBlur(overlay, (radius // 2 * 2 + 1, radius // 2 * 2 + 1), 0)

            alpha = 0.3
            background = background.astype(np.float32)
            background = (1 - alpha) * background + alpha * overlay
            background = np.clip(background, 0, 255).astype(np.uint8)

        return background
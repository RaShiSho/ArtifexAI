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
    print("Error: diffusers not installed. This package requires diffusers to work.")
    print("To install: pip install diffusers transformers accelerate")
    raise ImportError("diffusers package is required")


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

        print(f"Using device: {self.device}")

        # Initialize models
        if DIFFUSERS_AVAILABLE:
            self._load_stable_diffusion()
        else:
            raise RuntimeError("Required packages not available. Please install diffusers.")

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
            raise RuntimeError("Failed to load Stable Diffusion model")

    def generate_background(self, width, height, style='nature', seed=None, quality='medium'):
        """
        Generate high-quality background image using Stable Diffusion

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

        if self.stable_diffusion_pipe is None:
            raise RuntimeError("Stable Diffusion model not loaded")

        return self._generate_with_stable_diffusion(width, height, style, quality)

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
            raise RuntimeError("Failed to generate background with Stable Diffusion")

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
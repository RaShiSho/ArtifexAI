from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np

def pil_to_cv(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def cv_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def apply_named_filter(image, filter_name):
    if filter_name == 'none':
        return image

    elif filter_name == 'sharpen':
        return image.filter(ImageFilter.SHARPEN)

    elif filter_name == 'bright':
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.5)

    elif filter_name == 'smooth_skin':
        cv_img = pil_to_cv(image)
        result = cv2.bilateralFilter(cv_img, d=9, sigmaColor=75, sigmaSpace=75)
        return cv_to_pil(result)

    elif filter_name == 'glow':
        cv_img = pil_to_cv(image)
        blur = cv2.GaussianBlur(cv_img, (0, 0), sigmaX=15)
        glow = cv2.addWeighted(cv_img, 0.7, blur, 0.6, 0)
        return cv_to_pil(glow)

    elif filter_name == 'warm':
        r, g, b = image.split()
        r = r.point(lambda i: min(i + 30, 255))
        b = b.point(lambda i: max(i - 20, 0))
        return Image.merge('RGB', (r, g, b))

    elif filter_name == 'cold':
        r, g, b = image.split()
        b = b.point(lambda i: min(i + 30, 255))
        r = r.point(lambda i: max(i - 20, 0))
        return Image.merge('RGB', (r, g, b))


    elif filter_name == 'pencil':
        cv_img = pil_to_cv(image)
        dst_gray, _ = cv2.pencilSketch(cv_img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        return cv_to_pil(dst_gray)

    elif filter_name == 'hongkong':
        # 港风复古：增加黄色调和对比，降低饱和度
        img = image.convert("RGB")
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.7)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        r, g, b = img.split()
        r = r.point(lambda i: min(i + 10, 255))
        g = g.point(lambda i: min(i + 5, 255))
        return Image.merge("RGB", (r, g, b))

    elif filter_name == 'japan_soft':
        # 奶油日系：提亮 + 柔焦 + 粉调
        img = image.convert("RGB")
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.9)
        img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
        r, g, b = img.split()
        r = r.point(lambda i: min(i + 15, 255))
        b = b.point(lambda i: max(i - 10, 0))
        return Image.merge("RGB", (r, g, b))

    elif filter_name == 'golden_hour':
        # 黄金时刻（日落暖调光影）
        img = image.convert("RGB")
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)
        r, g, b = img.split()
        r = r.point(lambda i: min(i + 30, 255))
        g = g.point(lambda i: min(i + 15, 255))
        b = b.point(lambda i: max(i - 20, 0))
        return Image.merge("RGB", (r, g, b))

    elif filter_name == 'forest_film':
        # 清新自然（淡绿+柔光）
        img = image.convert("RGB")
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.85)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.05)
        r, g, b = img.split()
        g = g.point(lambda i: min(i + 20, 255))
        r = r.point(lambda i: max(i - 10, 0))
        return Image.merge("RGB", (r, g, b))

    elif filter_name == 'urban_black_gold':
        # 黑金城市风
        cv_img = pil_to_cv(image)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h = np.where((v > 50) & (s > 40), 30, h)  # 偏金色
        s = np.clip(s * 1.5, 0, 255).astype(np.uint8)
        v = np.clip(v * 1.1, 0, 255).astype(np.uint8)
        merged = cv2.merge([h, s, v])
        result = cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)
        return cv_to_pil(result)

    elif filter_name == 'retro_nostalgia':
        # 复古怀旧
        img = image.convert("RGB")
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.1)
        r, g, b = img.split()
        r = r.point(lambda i: min(i + 15, 255))
        g = g.point(lambda i: max(i - 5, 0))
        b = b.point(lambda i: max(i - 10, 0))
        return Image.merge("RGB", (r, g, b))


    else:
        return image

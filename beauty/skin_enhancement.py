import cv2
import numpy as np

class SkinEnhancer:
    """
    皮肤增强处理器
    提供肤色检测、肤色均衡、磨皮和美白的皮肤增强功能
    """
    def __init__(self):
        pass

    def detect_skin_mask(self, image):
        """
        检测皮肤区域并生成掩码
        参数:
            image: 输入BGR图像
        返回:
            numpy.ndarray: 皮肤区域二值掩码(0-255)
        """
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 定义皮肤色彩范围
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # 创建皮肤掩码
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 高斯模糊平滑掩码边缘(使过渡更自然)
        mask = cv2.GaussianBlur(mask, (3, 3), 1)

        return mask

    def balance_skin_tone(self, image, intensity=0.3):
        """
        肤色均衡处理 - 减少红色调并提亮肤色

        参数:
            image: 输入BGR图像
            intensity: 调节强度(0-1)

        返回:
            numpy.ndarray: 处理后的BGR图像
        """
        # 获取皮肤掩码
        skin_mask = self.detect_skin_mask(image)

        # 转换到LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 对皮肤区域进行色彩平衡
        # 减少红色调，增加亮度
        a_balanced = np.where(skin_mask > 0,
                              a * (1 - intensity * 0.1), a).astype(np.uint8)
        l_balanced = np.where(skin_mask > 0,
                              np.clip(l * (1 + intensity * 0.05), 0, 255), l).astype(np.uint8)

        # 合并通道
        lab_balanced = cv2.merge([l_balanced, a_balanced, b])

        # 转回BGR
        result = cv2.cvtColor(lab_balanced, cv2.COLOR_LAB2BGR)

        return result

    def skin_smoothing(self, image, intensity=0.8):
        """
               磨皮处理 - 平滑皮肤纹理同时保留边缘细节

               参数:
                   image: 输入BGR图像
                   intensity: 磨皮强度(0-1)

               返回:
                   numpy.ndarray: 处理后的BGR图像
        """
        # 获取皮肤掩码
        skin_mask = self.detect_skin_mask(image)

        # 双边滤波保持边缘
        smoothed = cv2.bilateralFilter(image, 15, 25, 25)

        # 高斯模糊进一步平滑
        blur = cv2.GaussianBlur(smoothed, (5, 5), 0)

        # 混合原图和模糊图
        smoothed = cv2.addWeighted(smoothed, 0.7, blur, 0.3, 0)

        # 创建3通道掩码
        skin_mask_3ch = cv2.merge([skin_mask, skin_mask, skin_mask]) / 255.0

        # 只对皮肤区域应用磨皮效果
        result = image * (1 - skin_mask_3ch * intensity) + smoothed * (skin_mask_3ch * intensity)

        return result.astype(np.uint8)

    def enhance_skin_brightness(self, image, brightness=1.1, contrast=1.05):
        """
                皮肤美白 - 提升皮肤区域亮度和对比度
                参数:
                    image: 输入BGR图像
                    brightness: 亮度增强系数(>1)
                    contrast: 对比度增强系数(>1)
                返回:
                    numpy.ndarray: 处理后的BGR图像
        """
        skin_mask = self.detect_skin_mask(image)

        # 转换到HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 增强亮度
        v_enhanced = np.where(skin_mask > 0,
                              np.clip(v * brightness, 0, 255), v).astype(np.uint8)

        # 微调饱和度
        s_enhanced = np.where(skin_mask > 0,
                              np.clip(s * contrast, 0, 255), s).astype(np.uint8)

        # 合并并转回BGR
        hsv_enhanced = cv2.merge([h, s_enhanced, v_enhanced])
        result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

        return result
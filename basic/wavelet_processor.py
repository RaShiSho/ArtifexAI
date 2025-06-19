import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt
import io
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import math
import pywt


class WaveletProcessor:

    @staticmethod
    def wavelet_transform(image_file, type='haar', level=1):
        try:
            img = WaveletProcessor._read_image(image_file, grayscale=False)
            level = int(level)

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            channels = cv2.split(img)
            cA_list = []
            visual_imgs = []

            for ch in channels:
                # 使用 wavedec2 计算多层系数
                coeffs = pywt.wavedec2(ch, wavelet=type, level=level)
                # 归一化系数，便于显示
                cA = coeffs[0] / np.abs(coeffs[0]).max()
                details = [tuple(d / np.abs(d).max() for d in detail) for detail in coeffs[1:]]

                # 使用 pywt 提供的 coeffs_to_array 把所有系数拼接为一个大数组
                arr, _ = pywt.coeffs_to_array([cA] + details)

                # 转为0~255 uint8灰度图
                arr_img = np.uint8(255 * (arr - arr.min()) / (arr.max() - arr.min()))

                visual_imgs.append(arr_img)
                # 用最后一级的近似系数转为图像（uint8）
                cA_img = np.uint8(255 * (cA - cA.min()) / (cA.max() - cA.min()))
                cA_list.append(cA_img)

            # 合并所有通道的近似系数图像
            result_img = cv2.merge(cA_list)
            # 合并所有通道的拼接系数图
            visual_img = cv2.merge(visual_imgs)

            # 转 base64
            result_base64 = WaveletProcessor._image_to_base64(result_img)
            visual_base64 = WaveletProcessor._image_to_base64(visual_img)

            return {
                "success": True,
                "resultImage": result_base64,
                "visualImage": visual_base64
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }




    @staticmethod
    def _read_image(image_file, grayscale=True):
        """读取图像文件并返回图像数组"""
        file_bytes = image_file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        if grayscale:
            return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        else:
            return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    @staticmethod
    def _image_to_base64(img):
        """将图像转换为base64编码"""
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode()
        return f"data:image/jpeg;base64,{img_str}"
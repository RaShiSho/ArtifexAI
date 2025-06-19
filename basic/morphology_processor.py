import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt
import io
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import math


class MorphologyProcessor:

    shape_map = {
        'rect': cv2.MORPH_RECT,
        'ellipse': cv2.MORPH_ELLIPSE,
        'cross': cv2.MORPH_CROSS
    }

    @staticmethod
    def morphology_processor(image_file, operation='Ideal', shape='cross', kernel_x=5, kernel_y=5):
        try:
            img = MorphologyProcessor._read_image(image_file, grayscale=False)

            cv_shape = MorphologyProcessor.shape_map.get(shape.lower(), cv2.MORPH_RECT)
            kernel = cv2.getStructuringElement(cv_shape, (int(kernel_x), int(kernel_y)))

            if operation.lower() == 'erosion':
                result = cv2.erode(img, kernel)

            elif operation.lower() == 'dilation':
                result = cv2.dilate(img, kernel)

            elif operation.lower() == 'closing':
                result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

            elif operation.lower() == 'opening':
                result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                
            else:
                raise ValueError(f"不支持的处理类型：{operation}，可选值为 'erosion', '', 'Gauss'。")

        
            # 将结果图转为 Base64
            result_base64 = MorphologyProcessor._image_to_base64(result)

            return {
                "success": True,
                "resultImage": result_base64
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
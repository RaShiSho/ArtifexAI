import cv2
import numpy as np
from io import BytesIO
import base64

class ResizeProcessor:
    @staticmethod
    def process_image(image_file, width, height, tx=0, ty=0, angle=0):
        try:
            # 读取图像文件为字节流
            file_bytes = image_file.read()
            # 将字节流转换为numpy数组
            nparr = np.frombuffer(file_bytes, np.uint8)
            # 解码图像
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 获取原始图像尺寸
            h, w = img.shape[:2]
            
            # 创建变换矩阵
            if angle != 0:
                # 旋转矩阵
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            else:
                # 平移矩阵
                M = np.float32([[1, 0, tx], [0, 1, ty]])
            
            # 应用仿射变换
            transformed = cv2.warpAffine(img, M, (w, h))
            
            # 计算缩放比例
            fx = width / transformed.shape[1]
            fy = height / transformed.shape[0]
            
            # 使用OpenCV的双线性插值进行缩放
            resized = cv2.resize(transformed, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
            
            # 将处理后的图像编码为JPEG格式
            _, buffer = cv2.imencode('.jpg', resized)
            img_str = base64.b64encode(buffer).decode()
            
            return {
                "success": True,
                "image": f"data:image/jpeg;base64,{img_str}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
import cv2
import numpy as np
import base64

class ArithmeticProcessor:
    @staticmethod
    def process_images(image_file1, image_file2, operation='add'):
        try:
            # 读取第一张图片
            file_bytes1 = image_file1.read()
            nparr1 = np.frombuffer(file_bytes1, np.uint8)
            img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
            
            # 读取第二张图片
            file_bytes2 = image_file2.read()
            nparr2 = np.frombuffer(file_bytes2, np.uint8)
            img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
            
            # 确保两张图片尺寸相同
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # 执行算术运算
            if operation == 'multiply':
                result = cv2.multiply(img1, img2)
            elif operation == 'subtract':
                result = cv2.subtract(img1, img2)
            else:  # 默认为加法
                result = cv2.add(img1, img2)
            
            # 将结果编码为base64
            _, buffer = cv2.imencode('.jpg', result)
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
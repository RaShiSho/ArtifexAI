import cv2
import numpy as np
import base64

class LogTransProcessor:
    @staticmethod
    def process_image(image_file, v):
        try:
            # 读取图片
            file_bytes = image_file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            # 计算变换参数 C
            v = float(v)
            C = 1.0 * v / np.log(1 + v)
            
            # 对图像进行对数变换
            result = C * np.log(1.0 + img)
            
            # 转换为8位无符号整数类型
            result = np.uint8(result)
            
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
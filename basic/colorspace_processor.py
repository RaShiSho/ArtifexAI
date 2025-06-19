import cv2
import numpy as np
import base64

class ColorSpaceProcessor:
    @staticmethod
    def process_image(image_file, analysis_type='rgb'):
        try:
            # 读取图片
            file_bytes = image_file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if analysis_type.lower() == 'rgb':
                # RGB分析
                channels = [img[:,:,0], img[:,:,1], img[:,:,2]]  # BGR顺序
                labels = ['B通道', 'G通道', 'R通道']
            else:
                # HSV分析
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                channels = [hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]]
                labels = ['H通道', 'S通道', 'V通道']
            
            # 转换每个通道为base64
            channel_images = []
            for channel in channels:
                _, buffer = cv2.imencode('.png', channel)
                img_str = base64.b64encode(buffer).decode()
                channel_images.append(f"data:image/png;base64,{img_str}")
            
            return {
                "success": True,
                "channels": channel_images,
                "labels": labels
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
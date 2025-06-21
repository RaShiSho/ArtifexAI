import torch

class YOLOPersonDetector:
    """使用YOLOv5检测图片中是否包含人的类"""
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.conf = 0.4  # 置信度阈值

    def detect_person(self, image_path):
        results = self.model(image_path)
        labels = results.pred[0][:, -1].cpu().numpy()
        # 获取类别名称映射表
        names = results.names
        # 检查是否有类别为'person'的检测结果
        return any(names[int(label)] == 'person' for label in labels)


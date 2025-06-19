
# detector_yolo.py
import torch

class YOLOPersonDetector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.conf = 0.4  # 置信度阈值

    def detect_person(self, image_path):
        results = self.model(image_path)
        labels = results.pred[0][:, -1].cpu().numpy()
        names = results.names
        return any(names[int(label)] == 'person' for label in labels)


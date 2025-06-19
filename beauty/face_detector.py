import cv2
import numpy as np


class FaceDetector:
    def __init__(self):
        # 使用OpenCV的Haar级联分类器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, image):
        """检测图片中的人脸"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # 扩展人脸区域以包含更多皮肤
        expanded_faces = []
        for (x, y, w, h) in faces:
            # 扩展边界框
            expand_ratio = 0.2
            new_x = max(0, int(x - w * expand_ratio))
            new_y = max(0, int(y - h * expand_ratio))
            new_w = min(image.shape[1] - new_x, int(w * (1 + 2 * expand_ratio)))
            new_h = min(image.shape[0] - new_y, int(h * (1 + 2 * expand_ratio)))

            expanded_faces.append((new_x, new_y, new_w, new_h))

        return expanded_faces

    def get_face_landmarks(self, image, face_box):
        """获取人脸关键点（简化版）"""
        x, y, w, h = face_box

        # 简单的关键点估计
        landmarks = {
            'left_eye': (x + w // 3, y + h // 3),
            'right_eye': (x + 2 * w // 3, y + h // 3),
            'nose': (x + w // 2, y + h // 2),
            'mouth': (x + w // 2, y + 2 * h // 3)
        }

        return landmarks
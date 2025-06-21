import cv2
import numpy as np

class FaceDetector:
    """人脸检测器类，用于检测图像中的人脸和关键点"""
    def __init__(self):
        """初始化人脸检测器，加载Haar级联分类器"""
        # 使用OpenCV的Haar级联分类器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, image):
        """
                检测图像中的人脸位置

                参数:
                    image: 输入图像(BGR格式)

                返回:
                    list: 扩展后的人脸边界框列表，每个元素为(x, y, w, h)
        """
        # 转换为灰度图像（人脸检测通常在灰度图上进行）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用级联分类器检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1, # 图像缩放比例，用于检测不同大小的人脸
            minNeighbors=5, # 每个候选矩形需要至少5个邻近矩形才能被保留
            minSize=(30, 30) # 最小人脸大小
        )

        # 扩展人脸区域以包含更多皮肤
        expanded_faces = []
        for (x, y, w, h) in faces:
            # 扩展边界框
            expand_ratio = 0.2
            # 计算扩展后的新坐标，确保不超出图像边界
            new_x = max(0, int(x - w * expand_ratio))
            new_y = max(0, int(y - h * expand_ratio))
            new_w = min(image.shape[1] - new_x, int(w * (1 + 2 * expand_ratio)))
            new_h = min(image.shape[0] - new_y, int(h * (1 + 2 * expand_ratio)))

            expanded_faces.append((new_x, new_y, new_w, new_h))

        return expanded_faces

    def get_face_landmarks(self, image, face_box):
        """
                估计人脸关键点
                参数:
                    image: 输入图像
                    face_box: 人脸边界框(x, y, w, h)
                返回:
                    dict: 关键点坐标字典，包含左右眼、鼻子和嘴巴位置
        """
        x, y, w, h = face_box

        # 基于人脸边界框的简单关键点估计
        landmarks = {
            'left_eye': (x + w // 3, y + h // 3),  # 左眼大约在1/3宽度，1/3高度处
            'right_eye': (x + 2 * w // 3, y + h // 3),  # 右眼大约在2/3宽度，1/3高度处
            'nose': (x + w // 2, y + h // 2),  # 鼻子在中心位置
            'mouth': (x + w // 2, y + 2 * h // 3)  # 嘴巴在中心位置，约在2/3高度处
        }

        return landmarks
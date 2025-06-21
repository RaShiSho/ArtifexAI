from .detector_yolo import YOLOPersonDetector
from .classifier_clip import CLIPClassifier
from .filters import apply_named_filter
from PIL import Image
import tempfile
import os
import base64
from io import BytesIO

isLoad = False

def classify_uploaded_image(pil_image, filter_map):
    global isLoad, yolo, clip_model

    if not isLoad:
        # 模型初始化（只加载一次）
        yolo = YOLOPersonDetector()
        clip_model = CLIPClassifier()
        isLoad = True


    # 保存图像到临时文件
    temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_path = temp.name
    pil_image.save(temp_path)
    temp.close()

    try:
        # 使用 YOLO 检测人
        if yolo.detect_person(temp_path):
            category = "人物"
        else:
            category = clip_model.classify(temp_path)

        # 获取类别对应的滤镜名
        filter_name = filter_map.get(category, "none")

        # 应用滤镜
        filtered_image = apply_named_filter(pil_image, filter_name)

        # 转 base64
        buffer = BytesIO()
        filtered_image.save(buffer, format="JPEG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "category": category,
            "image": base64_str
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

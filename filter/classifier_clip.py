import torch
import clip
from PIL import Image

class CLIPClassifier:
    """使用CLIP模型进行图像分类"""
    def __init__(self):
        # 加载CLIP模型和预处理方法
        self.model, self.preprocess = clip.load("ViT-B/32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # 类别定义和提示词
        self.categories = ["人物", "动物", "风景", "美食", "植物"]
        self.category_prompt_slices = [1, 1, 1, 1, 6]
        self.prompts = [
            "a person", "an animal", "a landscape", "delicious food",
            "green plants", "flowers", "a garden", "a bush with leaves", "a potted plant", "a tree"
        ]
        # 将提示词转换为模型可处理的token格式
        self.tokenized_prompts = clip.tokenize(self.prompts).to(self.device)

    def classify(self, image_path):
        """对输入图片进行分类"""
        # 预处理图片并转为tensor
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # 提取图片特征和文本特征
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(self.tokenized_prompts)
            # 计算相似度概率
            probs = (image_features @ text_features.T).softmax(dim=-1).squeeze()
        # 获取每个类别的最大概率
        group_scores = []
        idx = 0
        for count in self.category_prompt_slices:
            group_scores.append(probs[idx:idx+count].max().item())
            idx += count
        # 找到最大概率对应的类别
        best_category_idx = group_scores.index(max(group_scores))
        return self.categories[best_category_idx]

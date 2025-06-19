# classifier_clip.py
import torch
import clip
from PIL import Image

class CLIPClassifier:
    def __init__(self):
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
        self.tokenized_prompts = clip.tokenize(self.prompts).to(self.device)

    def classify(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(self.tokenized_prompts)
            probs = (image_features @ text_features.T).softmax(dim=-1).squeeze()

        group_scores = []
        idx = 0
        for count in self.category_prompt_slices:
            group_scores.append(probs[idx:idx+count].max().item())
            idx += count

        best_category_idx = group_scores.index(max(group_scores))
        return self.categories[best_category_idx]


import os
import shutil

def load_images_from_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def ensure_output_folders(base_folder, categories):
    os.makedirs(base_folder, exist_ok=True)
    for cat in categories:
        os.makedirs(os.path.join(base_folder, cat), exist_ok=True)

def move_to_category(image_path, category, output_folder):
    dest_folder = os.path.join(output_folder, category)
    shutil.copy(image_path, dest_folder)

import os
import uuid
import traceback
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image

# 原有美颜模块
from beauty.face_processor import FaceProcessor

# clarity 人脸增强模块
from clarity.face_enhance import FaceEnhancer

# 分类添加滤镜模块
from filter.classify import classify_uploaded_image
from filter.filters import apply_named_filter

# clarity 超分辨率增强模块
from clarity.realesrgan_enhancer import ImageEnhancer

# 背景替换模块
from background.main import BackgroundReplacementSystem
# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)

# 初始化模型
face_processor = FaceProcessor(model_path='clarity/models/GFPGANv1.4.pth', upscale=2)  # 美颜模块
face_enhancer = FaceEnhancer()
super_res_enhancer = ImageEnhancer(model_path='clarity/models/RealESRGAN_x4plus.pth')

# 新增：初始化背景替换系统
# 这将在服务器启动时执行一次，如果模型不存在，会自动下载
print("Initializing Background Replacement System... This may take a moment on the first run.")
background_system = BackgroundReplacementSystem()
print("Background Replacement System initialized successfully.")

# 创建必要目录
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 页面路由
@app.route('/')
def homepage():
    return render_template("home.html")

@app.route('/filters')
def filters():
    return render_template("index.html")

@app.route('/beauty')
def beauty_page():
    return render_template("beauty.html")

@app.route('/clarity')
def clarity_page():
    return render_template("clarity.html")

@app.route('/background')
def background_page():
    return render_template("background.html")

# 滤镜处理接口
@app.route('/apply_filter_custom', methods=['POST'])
def apply_filter_custom():
    images = request.files.getlist('images')
    filter_map = {
        '人物': request.form.get('filter_map[人物]', 'none'),
        '动物': request.form.get('filter_map[动物]', 'none'),
        '风景': request.form.get('filter_map[风景]', 'none'),
        '美食': request.form.get('filter_map[美食]', 'none'),
        '植物': request.form.get('filter_map[植物]', 'none'),
    }

    results = []
    for image_file in images:
        image = Image.open(image_file.stream).convert("RGB")
        result = classify_uploaded_image(image, filter_map)
        results.append({
            "category": result["category"],
            "filter": filter_map[result["category"]],
            "filtered_image_base64": result["image"]
        })

    return jsonify({"results": results})

# 美颜接口
@app.route('/api/beauty', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': '未检测到图片'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400

        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{filename}")
        file.save(input_path)

        output_filename = f"result_{os.path.basename(input_path)}"
        output_path = os.path.join(RESULT_FOLDER, output_filename)

        face_processor.process_face(input_path, output_path)

        return send_file(output_path, mimetype='image/jpeg')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

# 新增 clarity 模块接口：人脸增强 + 背景超分
@app.route('/api/face-enhance', methods=['POST'])
def api_face_enhance():
    try:
        if 'image' not in request.files:
            return jsonify({'error': '未检测到图片'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400

        # 保存上传图像
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{filename}")
        file.save(input_path)

        # 读取为 OpenCV 格式图像
        input_img = cv2.imdecode(np.fromfile(input_path, np.uint8), cv2.IMREAD_COLOR)
        if input_img is None:
            return jsonify({'error': '图像读取失败'}), 400

        # 执行人脸增强
        output_img = face_enhancer.enhance(input_img)

        # 保存输出图像
        output_path = os.path.join(RESULT_FOLDER, f"enhanced_{os.path.basename(filename)}")
        cv2.imwrite(output_path, output_img)

        return send_file(output_path, mimetype='image/jpeg')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500


@app.route('/api/super-resolution', methods=['POST'])
def api_super_resolution():
    try:
        if 'image' not in request.files:
            return jsonify({'error': '未检测到图片'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400

        # 保存上传图像
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{filename}")
        file.save(input_path)

        # 定义输出路径
        output_filename = f"sr_{os.path.basename(filename)}"
        output_path = os.path.join(RESULT_FOLDER, output_filename)

        # 调用 ImageEnhancer 的 enhance 方法进行超分处理
        # 该方法会读取 input_path 的图像，并将结果保存到 output_path
        super_res_enhancer.enhance(input_path, output_path)

        # 将处理后的文件发送回客户端
        return send_file(output_path, mimetype='image/jpeg')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500


@app.route('/api/replace-background', methods=['POST'])
def api_replace_background():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file found'}), 400
        if 'theme' not in request.form:
            return jsonify({'error': 'No theme selected'}), 400

        file = request.files['image']
        theme = request.form['theme']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # 保存上传图片
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{filename}")
        file.save(input_path)

        # 定义输出路径
        output_filename = f"bg_replaced_{os.path.basename(filename)}"
        output_path = os.path.join(RESULT_FOLDER, output_filename)

        # 使用预先初始化的模型处理图片
        background_system.process_image(
            input_path=input_path,
            output_path=output_path,
            background_style=theme,
            quality='high'  # 默认为高质量以获得更好效果
        )

        return send_file(output_path, mimetype='image/jpeg')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


# 启动服务
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

import os
import uuid
import traceback
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"
# 基本图像处理模块
from basic.resize_processor import ResizeProcessor
from basic.colorspace_processor import ColorSpaceProcessor
from basic.arithmetic_processor import ArithmeticProcessor
from basic.logtrans_processor import LogTransProcessor
from basic.histogram_processor import HistogramProcessor
from basic.segmentation_processor import SegmentationProcessor
from basic.smooth_processor import SmoothProcessor
from basic.sharpen_processor import SharpenProcessor
from basic.morphology_processor import MorphologyProcessor
from basic.restore_processor import RestoreProcessor
from basic.wavelet_processor import WaveletProcessor

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


@app.route('/basic')
def basic_page():
    return render_template('basic.html')

@app.route('/resize')
def resize_page():
    return render_template('resize.html')

@app.route('/colorspace')
def colorspace_page():
    return render_template('colorspace.html')

@app.route('/arithmetic')
def arithmetic():
    return render_template('arithmetic.html')

@app.route('/logtrans')
def logtrans():
    return render_template('logtrans.html')

@app.route('/histogram')
def histogram():
    return render_template('histogram.html')

@app.route('/histogram_normalize')
def histogram_normalize():
    return render_template('histogram_normalize.html')

@app.route('/segmentation')
def segmentation():
    return render_template('segmentation.html')

@app.route('/smooth')
def smooth():
    return render_template('smooth.html')

@app.route('/sharpen')
def sharpen():
    return render_template('sharpen.html')

@app.route('/morphology')
def morphology():
    return render_template('morphology.html')

@app.route('/restore')
def restore():
    return render_template('restore.html')

@app.route('/wavelettrans')
def wavelettrans():
    return render_template('wavelettrans.html')


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

        # 验证主题是否有效
        try:
            available_styles = background_system.get_available_styles()
            if theme not in available_styles:
                return jsonify({'error': f'Invalid theme. Available themes: {available_styles}'}), 400
        except Exception as e:
            return jsonify({'error': f'Failed to get available styles: {str(e)}'}), 500

        # 保存上传图片
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{filename}")
        file.save(input_path)

        # 定义输出路径
        output_filename = f"bg_replaced_{os.path.basename(filename)}"
        output_path = os.path.join(RESULT_FOLDER, output_filename)

        # 使用背景替换系统处理图片
        background_system.process_image(
            input_path=input_path,
            output_path=output_path,
            background_style=theme,
            enhance_edges=True,
            feather_strength=3,
            quality='high'  # 默认为高质量以获得更好效果
        )

        return send_file(output_path, mimetype='image/jpeg')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# 获取可用背景风格接口
@app.route('/api/background-styles', methods=['GET'])
def api_get_background_styles():
    try:
        styles = background_system.get_available_styles()
        return jsonify({'styles': styles})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Failed to get styles: {str(e)}'}), 500



# 图像缩放与变换 路由
@app.route('/resize', methods=['POST'])
def resize_image():
    if 'image' not in request.files:
        return jsonify({"error": "没有上传图片"}), 400
    
    try:
        # 获取所有参数
        width = int(request.form.get('width', 800))
        height = int(request.form.get('height', 600))
        tx = int(request.form.get('tx', 0))  # 水平平移
        ty = int(request.form.get('ty', 0))  # 垂直平移
        angle = float(request.form.get('angle', 0))  # 旋转角度
        
        # 调用处理函数
        result = ResizeProcessor.process_image(
            request.files['image'],
            width,
            height,
            tx=tx,
            ty=ty,
            angle=angle
        )
        
        if result["success"]:
            return jsonify(result)
        else:
            return jsonify({"error": result["error"]}), 400
    except ValueError as e:
        return jsonify({"error": "参数值无效"}), 400


# 色彩空间分析 路由
@app.route('/analyze_colorspace', methods=['POST'])
def analyze_colorspace():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    analysis_type = request.form.get('type', 'rgb')
    result = ColorSpaceProcessor.process_image(request.files['image'], analysis_type)
    return jsonify(result)


# 图像算术运算 路由
@app.route('/arithmetic', methods=['POST'])
def process_arithmetic():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "请上传两张图片"}), 400
    
    operation = request.form.get('operation', 'add')
    result = ArithmeticProcessor.process_images(
        request.files['image1'],
        request.files['image2'],
        operation
    )
    
    return jsonify(result)


# 对数变换 路由
@app.route('/logtrans', methods=['POST'])
def process_logtrans():
    if 'image' not in request.files:
        return jsonify({"error": "请上传图片"}), 400
    
    v = request.form.get('v', '100')
    result = LogTransProcessor.process_image(
        request.files['image'],
        v
    )
    
    return jsonify(result)




# 直方图处理 路由
@app.route('/generate_histogram', methods=['POST'])
def generate_histogram():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    histogram_type = request.form.get('type', 'gray')
    result = HistogramProcessor.process_image(request.files['image'], histogram_type)
    return jsonify(result)

@app.route('/equalize_histogram', methods=['POST'])
def equalize_histogram():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    result = HistogramProcessor.equalize_histogram(request.files['image'])
    return jsonify(result)

@app.route('/linear_transform_histogram', methods=['POST'])
def linear_transform_histogram():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    result = HistogramProcessor.linear_transform_histogram(request.files['image'])
    return jsonify(result)

@app.route('/normalize_histogram', methods=['POST'])
def normalize_histogram():
    if 'source_image' not in request.files or 'target_image' not in request.files:
        return jsonify({"success": False, "error": "请上传原图像和目标图像"})
    
    result = HistogramProcessor.normalize_histogram(
        request.files['source_image'],
        request.files['target_image']
    )
    return jsonify(result)



# 图像分割 路由
@app.route('/basic_enhance_detail', methods=['POST'])
def basic_enhance_detail():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    result = SegmentationProcessor.basic_enhance_detail(request.files['image'])
    return jsonify(result)

@app.route('/edge_detection', methods=['POST'])
def edge_detection():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    operator = request.form.get('operator', 'roberts')
    result = SegmentationProcessor.edge_detection(request.files['image'], operator)
    return jsonify(result)

@app.route('/line_change_detection', methods=['POST'])
def line_change_detection():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    result = SegmentationProcessor.line_change_detection(request.files['image'])
    return jsonify(result)


# 图像平滑 路由
@app.route('/frequency_smoothing', methods=['POST'])
def frequency_smoothing():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    filter = request.form.get('filter', 'Ideal')
    result = SmoothProcessor.frequency_domain_smoothing(request.files['image'], filter)
    return jsonify(result)

@app.route('/spatial_smoothing', methods=['POST'])
def spatial_smoothing():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    filter = request.form.get('filter', 'Mean')
    result = SmoothProcessor.spatial_domain_smoothing(request.files['image'], filter)
    return jsonify(result)


# 图像锐化 路由
@app.route('/frequency_sharpening', methods=['POST'])
def frequency_sharpening():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    filter = request.form.get('filter', 'Ideal')
    result = SharpenProcessor.frequency_domain_sharpening(request.files['image'], filter)
    return jsonify(result)

@app.route('/spatial_sharpening', methods=['POST'])
def spatial_sharpening():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    operator = request.form.get('operator', 'Roberts')
    result = SharpenProcessor.spatial_domain_sharpening(request.files['image'], operator)
    return jsonify(result)


# 数学形态学操作 路由
@app.route('/morphology_processor', methods=['POST'])
def morphology_processor():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})

    operation = request.form.get('operation', 'erosion')
    shape = request.form.get('shape', 'cross')
    kernel_x = request.form.get('kernel_x', '5')
    kernel_y = request.form.get('kernel_y', '5')
    result = MorphologyProcessor.morphology_processor(
        request.files['image'], 
        operation, 
        shape, 
        kernel_x, 
        kernel_y
    )
    return jsonify(result)


# 图像恢复 路由
@app.route('/add_noise', methods=['POST'])
def add_noise():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})

    noise_type = request.form.get('noise_type', 'saltpepper')
    result = RestoreProcessor.add_noise(request.files['image'], noise_type)
    return jsonify(result)

@app.route('/mean_filter', methods=['POST'])
def mean_filter():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})

    filter_x = request.form.get('filter_x', 3)
    filter_y = request.form.get('filter_y', 3)
    result = RestoreProcessor.meanFiltering(request.files['image'], filter_x, filter_y)
    return jsonify(result)

@app.route('/statistical_filter', methods=['POST'])
def statistical_filter():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})

    filter_x = request.form.get('filter_x', 3)
    filter_y = request.form.get('filter_y', 3)
    type = request.form.get('type', 'median')
    result = RestoreProcessor.statisticalFiltering(request.files['image'], type, filter_x, filter_y)
    return jsonify(result)

@app.route('/selective_filter', methods=['POST'])
def selective_filter():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})

    filter_x = request.form.get('filter_x', 3)
    filter_y = request.form.get('filter_y', 3)
    type = request.form.get('type', 'bandPass') 
    up = request.form.get('up', 220)
    down = request.form.get('down', 20)
    result = RestoreProcessor.selectiveFiltering(request.files['image'], up, down, type)
    return jsonify(result)



# 小波变换 路由
@app.route('/wavelet_transform', methods=['POST'])
def wavelet_transform():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})

    type = request.form.get('type', 'haar')
    level = request.form.get('level', 1)
    result = WaveletProcessor.wavelet_transform(request.files['image'], type, level)
    return jsonify(result)

# 启动服务
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt
import io
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

class HistogramProcessor:
    @staticmethod
    def process_image(image_file, histogram_type='gray'):
        try:
            if histogram_type.lower() == 'rgb':
                # 读取彩色图像
                img = HistogramProcessor._read_image(image_file, grayscale=False)
                
                # 计算RGB各通道直方图
                r_hist = cv2.calcHist([img], [2], None, [256], [0, 256])  # 红色通道
                g_hist = cv2.calcHist([img], [1], None, [256], [0, 256])  # 绿色通道
                b_hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # 蓝色通道
                
                # 创建RGB直方图图表
                HistogramProcessor._create_histogram_plot(
                    [r_hist, g_hist, b_hist], 
                    ['red', 'green', 'blue'],
                    with_legend=True
                )
            
            else:
                # 读取灰度图像
                img = HistogramProcessor._read_image(image_file)
                
                # 计算灰度直方图
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                
                # 绘制灰度直方图
                HistogramProcessor._create_histogram_plot(hist)
            
            # 转换直方图为base64
            histogram_image = HistogramProcessor._plot_to_base64()

            # 转换原图为base64
            original_image = HistogramProcessor._image_to_base64(img)
            
            return {
                "success": True,
                "original": original_image,
                "histogram": histogram_image
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def equalize_histogram(image_file):
        try:
            # 读取灰度图像
            gray_img = HistogramProcessor._read_image(image_file)
            
            # 灰度图均衡化
            equalized_img = cv2.equalizeHist(gray_img)
            
            # 计算原始和均衡化后的灰度直方图
            orig_hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
            equalized_hist = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])
            
            # 创建原始灰度直方图图表
            HistogramProcessor._create_histogram_plot(orig_hist)
            orig_hist_image = HistogramProcessor._plot_to_base64()
            
           # 创建均衡化后的灰度直方图图表
            HistogramProcessor._create_histogram_plot(equalized_hist)
            equalized_hist_image = HistogramProcessor._plot_to_base64()
            
            # 转换原始和均衡化后的灰度图为base64
            original_image = HistogramProcessor._image_to_base64(gray_img)
            equalized_image = HistogramProcessor._image_to_base64(equalized_img)
            
            
            return {
                "success": True,
                "original": original_image,
                "equalized": equalized_image,
                "original_histogram": orig_hist_image,
                "equalized_histogram": equalized_hist_image
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # 在文件末尾添加以下方法
    
    @staticmethod
    def linear_transform_histogram(image_file):
        try:
            # 读取灰度图像
            gray_img = HistogramProcessor._read_image(image_file)
            
            # 创建输出图像
            h, w = gray_img.shape[:2]
            transformed_img = np.zeros(gray_img.shape, np.float64)
            
            # 通过遍历对不同像素范围内进行分段线性变化
            for i in range(h):
                for j in range(w):
                    pix = gray_img[i][j]
                    if pix < 70:
                        transformed_img[i][j] = 0.8 * pix
                    elif pix < 180:
                        transformed_img[i][j] = 2.6 * pix - 210
                    else:
                        transformed_img[i][j] = 0.138 * pix + 195
            
            # 四舍五入并转换为uint8类型
            transformed_img = np.around(transformed_img)
            transformed_img = transformed_img.astype(np.uint8)
            
            # 计算原始和变换后的灰度直方图
            orig_hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
            transformed_hist = cv2.calcHist([transformed_img], [0], None, [256], [0, 256])
            
            # 创建原始灰度直方图图表
            HistogramProcessor._create_histogram_plot(orig_hist)
            orig_hist_image = HistogramProcessor._plot_to_base64()
            
            # 创建变换后的灰度直方图图表
            HistogramProcessor._create_histogram_plot(transformed_hist)
            transformed_hist_image = HistogramProcessor._plot_to_base64()
            
            # 转换原始和变换后的灰度图为base64
            original_image = HistogramProcessor._image_to_base64(gray_img)
            transformed_image = HistogramProcessor._image_to_base64(transformed_img)
            
            return {
                "success": True,
                "original": original_image,
                "transformed": transformed_image,
                "original_histogram": orig_hist_image,
                "transformed_histogram": transformed_hist_image
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    def normalize_histogram(source_file, target_file):
        try:
            # 读取原图像和目标图像
            source_img = HistogramProcessor._read_image(source_file)
            target_file.seek(0)  # 重置文件指针
            target_img = HistogramProcessor._read_image(target_file)
            
            # 计算原图像直方图
            source_hist = np.zeros(256, dtype=np.int32)
            h, w = source_img.shape
            for i in range(h):
                for j in range(w):
                    source_hist[source_img[i, j]] += 1
            
            # 计算目标图像直方图
            target_hist = np.zeros(256, dtype=np.int32)
            th, tw = target_img.shape
            for i in range(th):
                for j in range(tw):
                    target_hist[target_img[i, j]] += 1
            
            # 计算原图像的累积分布函数
            source_cdf = np.zeros(256, dtype=np.float64)
            source_cdf[0] = source_hist[0] / source_img.size
            for i in range(1, 256):
                source_cdf[i] = source_cdf[i-1] + source_hist[i] / source_img.size
            
            # 计算目标图像的累积分布函数
            target_cdf = np.zeros(256, dtype=np.float64)
            target_cdf[0] = target_hist[0] / target_img.size
            for i in range(1, 256):
                target_cdf[i] = target_cdf[i-1] + target_hist[i] / target_img.size
            
            # 将累积分布函数映射到 0-255
            source_cdf_mapped = np.round(source_cdf * 255).astype(np.uint8)
            target_cdf_mapped = np.round(target_cdf * 255).astype(np.uint8)
            
            # 创建映射表
            mapping = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                a = source_cdf_mapped[i]
                min_diff = 255
                min_j = 0
                for j in range(256):
                    diff = abs(int(target_cdf_mapped[j]) - int(a))
                    if diff < min_diff:
                        min_diff = diff
                        min_j = j
                mapping[i] = min_j
            
            # 应用映射到原图像
            normalized_img = np.zeros_like(source_img)
            for i in range(h):
                for j in range(w):
                    normalized_img[i, j] = mapping[source_img[i, j]]
            
            # 计算原始灰度直方图
            source_hist_cv = cv2.calcHist([source_img], [0], None, [256], [0, 256])
            
            # 计算目标灰度直方图
            target_hist_cv = cv2.calcHist([target_img], [0], None, [256], [0, 256])
            
            # 计算正规化后的灰度直方图
            normalized_hist = cv2.calcHist([normalized_img], [0], None, [256], [0, 256])
            
            # 创建原始灰度直方图图表
            HistogramProcessor._create_histogram_plot(source_hist_cv)
            source_hist_image = HistogramProcessor._plot_to_base64()
            
            # 创建目标灰度直方图图表
            HistogramProcessor._create_histogram_plot(target_hist_cv)
            target_hist_image = HistogramProcessor._plot_to_base64()
            
            # 创建正规化后的灰度直方图图表
            HistogramProcessor._create_histogram_plot(normalized_hist)
            normalized_hist_image = HistogramProcessor._plot_to_base64()
            
            # 转换原始灰度图为base64
            source_image = HistogramProcessor._image_to_base64(source_img)
            
            # 转换目标灰度图为base64
            target_image = HistogramProcessor._image_to_base64(target_img)
            
            # 转换正规化后的灰度图为base64
            normalized_image = HistogramProcessor._image_to_base64(normalized_img)
            
            return {
                "success": True,
                "source": source_image,
                "target": target_image,
                "normalized": normalized_image,
                "source_histogram": source_hist_image,
                "target_histogram": target_hist_image,
                "normalized_histogram": normalized_hist_image
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


    @staticmethod
    def _read_image(image_file, grayscale=True):
        """读取图像文件并返回图像数组"""
        file_bytes = image_file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        if grayscale:
            return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        else:
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    @staticmethod
    def _create_histogram_plot(hist_data, color='gray', title=None, with_legend=False):
        """创建直方图图表"""
        plt.figure(figsize=(10, 6))
        
        if isinstance(hist_data, list) and isinstance(color, list):
            # 多通道直方图
            for i, (hist, clr, label) in enumerate(zip(hist_data, color, ['Red', 'Green', 'Blue'])):
                plt.plot(hist, color=clr, label=label)
            if with_legend:
                plt.legend()
        else:
            # 单通道直方图
            plt.plot(hist_data, color=color)
        
        plt.xlabel('pixel')
        plt.ylabel('frequency')
        plt.xlim([0, 256])
        if title:
            plt.title(title)
    
    @staticmethod
    def _plot_to_base64():
        """将当前图表转换为base64编码的图像"""
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        return f"data:image/png;base64,{img_str}"

    @staticmethod
    def _image_to_base64(img):
        """将图像转换为base64编码"""
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode()
        return f"data:image/jpeg;base64,{img_str}"
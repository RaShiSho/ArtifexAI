import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt
import io
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

class SegmentationProcessor:
    @staticmethod
    def basic_enhance_detail(image_file):
        try:
            # 读取灰度图像
            gray_img = SegmentationProcessor._read_image(image_file)
            
            gray_img = gray_img.astype('float')
            row, col = gray_img.shape
            gradient = np.zeros((row, col))

            for x in range(row - 1):
                for y in range(col - 1):
                    gx = abs(gray_img[x + 1, y] - gray_img[x, y])
                    gy = abs(gray_img[x, y + 1] - gray_img[x, y])
                    gradient[x, y] = gx + gy
            sharp = gray_img + gradient

            sharp = np.where(sharp > 255, 255, sharp)
            sharp = np.where(sharp < 0, 0, sharp)
            
            # 转换原始和均衡化后的灰度图为base64
            gray_image = SegmentationProcessor._image_to_base64(gray_img)
            gradient_image = SegmentationProcessor._image_to_base64(gradient)
            enhanced_image = SegmentationProcessor._image_to_base64(sharp)
            
            return {
                "success": True,
                "originalGrayImage": gray_image,
                "gradientImage": gradient_image,
                "enhancedGrayImage": enhanced_image
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    def edge_detection(image_file, operator='Roberts'):
        try:
            # 转为灰度图
            gray = SegmentationProcessor._read_image(image_file)

            if operator.lower() == 'roberts':
                # Roberts 算子
                kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
                kernely = np.array([[0, -1], [1, 0]], dtype=int)
                x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
                y = cv2.filter2D(gray, cv2.CV_16S, kernely)
                absX = cv2.convertScaleAbs(x)
                absY = cv2.convertScaleAbs(y)
                result = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

            elif operator.lower() == 'sobel':
                # Sobel 算子
                grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
                grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
                absX = cv2.convertScaleAbs(grad_x)
                absY = cv2.convertScaleAbs(grad_y)
                result = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

            elif operator.lower() == 'laplacian':
                # Laplacian 算子
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                dst = cv2.Laplacian(blurred, cv2.CV_16S, ksize=3)
                result = cv2.convertScaleAbs(dst)

            elif operator.lower() == 'log':
                # LoG 算子
                # 先对原图做边缘扩充再高斯模糊
                extended = cv2.copyMakeBorder(gray, 2, 2, 2, 2, borderType=cv2.BORDER_REPLICATE)
                blurred = cv2.GaussianBlur(extended, (3, 3), 0, 0)
                # 定义 LoG 核
                m1 = np.array([
                    [0, 0, -1, 0, 0],
                    [0, -1, -2, -1, 0],
                    [-1, -2, 16, -2, -1],
                    [0, -1, -2, -1, 0],
                    [0, 0, -1, 0, 0]
                ])

                r, c = blurred.shape
                image1 = np.zeros(blurred.shape)

                
                for i in range(2, r - 2):
                    for j in range(2, c - 2):
                        image1[i, j] = np.sum((m1 * blurred[i - 2:i + 3, j - 2:j + 3]))

                result = cv2.convertScaleAbs(image1)

                    # 为了对每个通道都进行卷积，需要分通道处理，再合并
                    # channels = cv2.split(blurred)
                    # filtered_channels = []
                    # for ch in channels:
                    #     ch = ch.astype(np.int32)
                    #     r, c = ch.shape
                    #     temp = np.zeros_like(ch)
                    #     for i in range(2, r - 2):
                    #         for j in range(2, c - 2):
                    #             region = ch[i - 2:i + 3, j - 2:j + 3]
                    #             temp[i, j] = np.sum(region * m1)
                    #     filtered_channels.append(cv2.convertScaleAbs(temp))
                    # result = cv2.merge(filtered_channels)

            else:
                raise ValueError(f"不支持的算子类型：{operator}，可选值为 'roberts', 'sobel', 'laplacian', 'log'。")

            # 将结果图转为 Base64
            gray_image = SegmentationProcessor._image_to_base64(gray)
            result_base64 = SegmentationProcessor._image_to_base64(result)

            return {
                "success": True,
                "originalGrayImage": gray_image,
                "edgeImage": result_base64
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


    @staticmethod
    def line_change_detection(image_file):
        try:
            img = SegmentationProcessor._read_image(image_file, grayscale=False)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.GaussianBlur(img, (3, 3), 0)
            edges = cv2.Canny(img, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
            result = img.copy()
            # print(lines)
            for i_line in lines:
                for line in i_line:
                    rho = line[0]  # 第一个元素是距离rho
                    theta = line[1]  # 第二个元素是角度theta
                    if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
                        # 该直线与第一行的交点
                        pt1 = (int(rho / np.cos(theta)), 0)
                        # 该直线与最后一行的焦点
                        pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                        # 绘制一条红线
                        cv2.line(result, pt1, pt2, (0, 0, 255))
                    else:  # 水平直线
                        # 该直线与第一列的交点
                        pt1 = (0, int(rho / np.sin(theta)))
                        # 该直线与最后一列的交点
                        pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                        # 绘制一条直线
                        cv2.line(result, pt1, pt2, (0, 0, 255), 1)
            # 经验参数
            minLineLength = 80
            maxLineGap = 15
            linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 80)

            result_P = img.copy()
            for i_P in linesP:
                for x1, y1, x2, y2 in i_P:
                    cv2.line(result_P, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # 将结果图转为 Base64
            gray_image = SegmentationProcessor._image_to_base64(gray)
            result_base64 = SegmentationProcessor._image_to_base64(result)
            result_P_base64 = SegmentationProcessor._image_to_base64(result_P)

            return {
                "success": True,
                "originalGrayImage": gray_image,
                "lineImage": result_base64,
                "linePImage": result_P_base64
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
# ArtifexAI - 智能图像处理平台

## 项目简介

ArtifexAI 是一个基于 Flask 框架构建的智能图像处理平台，旨在提供一系列先进的图像处理功能。它集成了多种深度学习模型和传统图像处理算法，为用户提供美颜、人脸增强、超分辨率、智能背景替换以及基于内容的图像滤镜分类等服务。本平台致力于简化复杂的图像处理流程，让用户能够轻松实现高质量的图像编辑和创作。

## 主要功能

ArtifexAI 提供了以下核心功能：

*   **美颜与人脸增强**：利用先进的算法对人脸进行美化，包括磨皮、肤色调整、面部特征优化等，并能对人脸细节进行超分辨率增强。
*   **智能背景替换**：通过精确的图像分割技术，实现人像与背景的分离，并支持智能生成或替换为多种风格的背景（如自然、城市、抽象、奇幻、太空、复古等）。
*   **超分辨率增强**：应用深度学习模型（如 Real-ESRGAN）提升图像分辨率，恢复图像细节，使低分辨率图像变得清晰。
*   **图像内容分类与滤镜应用**：结合 CLIP 和 YOLOv5 等模型，智能识别图像内容，并根据内容自动推荐或应用合适的图像滤镜。
*   **基础图像处理**：包含图像缩放、色彩空间转换、算术运算、对数变换、直方图处理（均衡化、归一化）、图像分割（边缘检测）、图像平滑与锐化、形态学处理以及图像恢复等传统图像处理功能。

## 部署方式

本项目推荐使用 Python 虚拟环境进行部署，以管理项目依赖。

1.  **克隆仓库**：
    ```bash
    git clone https://github.com/RaShiSho/ArtifexAI.git
    cd ArtifexAI
    ```

2.  **创建并激活虚拟环境**：
    ```bash
    uv venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **安装依赖**：

    **大部分依赖**:
	```bash
    uv pip install -r requirements.txt
    ```
    
    *部分依赖（如 `torch`、`torchvision`）可能需要根据您的 CUDA 版本手动安装。请参考以下命令进行安装。*
    1. 删除按照 requirements.txt 下载的 torch 库
    ```bash
    uv pip uninstall torch torchvision torchaudio
	```
	2. 通过官网链接下载 wheel 文件安装包
	```bash
	uv pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
	```

4. **部分包文件修改**
	由于我们使用的库的版本更新问题，我们需要手动修改库中的文件
	定位到 `ArtifexAI\.venv\Lib\site-packages\basicsr\data\degradations.py` 文件下
	将第 8 行的
	```python
	from torchvision.transforms.functional_tensor import rgb_to_grayscale
	```
	修改为
	```python
	from torchvision.transforms._functional_tensor import rgb_to_grayscale
	```
	

5.  **模型下载**：
    *   本项目中的部分模型（如 Stable Diffusion、DeepLabV3）会在首次运行时自动下载到您的本地缓存目录（通常是 `~/.cache/huggingface/hub/` 和 `~/.cache/torch/hub/checkpoints/`）。请确保您的网络连接正常。
    *   如果您遇到 `clip` 库的 `AttributeError: module 'clip' has no attribute 'load'` 错误，请尝试重新安装 `clip` 库：`pip install --upgrade clip` 或 `pip install git+https://github.com/openai/CLIP.git`。

## 运行项目

在完成部署后，您可以通过以下命令运行 Flask 应用：

1.  **激活虚拟环境**（如果尚未激活）：
    ```bash
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

2.  **运行 Flask 应用**：
    ```bash
    python app.py
    ```

3.  **访问应用**：
    应用程序启动后，您将在终端看到类似以下输出：
    ```
     * Running on http://127.0.0.1:5000
    ```
    在您的浏览器中打开 `http://127.0.0.1:5000` 即可访问 ArtifexAI 平台。
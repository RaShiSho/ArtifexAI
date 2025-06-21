"""
Real-ESRGAN 图像超分辨率增强实现
包含网络结构和图像处理流程
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class RRDBBlock(nn.Module):
    """残差中的残差密集块 (Residual in Residual Dense Block)"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        """
               初始化RRDB块
               参数:
                   num_feat: 特征图通道数
                   num_grow_ch: 每层增长通道数
        """
        super(RRDBBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        """前向传播"""
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class ResidualDenseBlock(nn.Module):
    """残差密集块 (Residual Dense Block)"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        """
               初始化RDB块
               参数:
                   num_feat: 特征图通道数
                   num_grow_ch: 每层增长通道数
        """
        super(ResidualDenseBlock, self).__init__()
        # 5层卷积，每层输入都是前面所有层的输出拼接
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RealESRGANer(nn.Module):
    """RealESRGAN网络结构"""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        """
               初始化RealESRGAN
               参数:
                   num_in_ch: 输入通道数
                   num_out_ch: 输出通道数
                   num_feat: 特征图通道数
                   num_block: RRDB块数量
                   num_grow_ch: 每层增长通道数
                   scale: 放大倍数
        """
        super(RealESRGANer, self).__init__()
        self.scale = scale
        self.num_feat = num_feat

        # 第一层卷积
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # RRDB块
        self.body = nn.ModuleList()
        for _ in range(num_block):
            self.body.append(RRDBBlock(num_feat, num_grow_ch))

        # 主干网络后的卷积
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # 上采样层
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """前向传播"""
        # 初始卷积
        feat = self.conv_first(x)
        body_feat = feat
        # 通过所有RRDB块
        for block in self.body:
            body_feat = block(body_feat)
        # 主干网络后的卷积和残差连接
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat

        # 上采样
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class ImageEnhancer:
    """图像增强处理器"""
    def __init__(self, model_path, scale=4, tile_size=0, tile_pad=10, pre_pad=0, half=False):
        """
                初始化增强器
                参数:
                    model_path: 模型路径
                    scale: 放大倍数
                    tile_size: 分块大小(0表示不分块)
                    tile_pad: 分块填充
                    pre_pad: 预处理填充
                    half: 是否使用半精度
        """
        # 自动选择设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scale = scale
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.half = half

        # 加载模型
        self.model = RealESRGANer(scale=scale)
        loadnet = torch.load(model_path, map_location=self.device)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        self.model.load_state_dict(loadnet[keyname], strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)
        if self.half:
            self.model = self.model.half()

    def pre_process(self, img):
        """预处理图像"""
        img = img.astype(np.float32)
        # 处理16位和8位图像
        if np.max(img) > 256:
            max_range = 65535
        else:
            max_range = 255
        img = img / max_range
        # 处理灰度图和RGBA图像
        if len(img.shape) == 2:  # 灰度图
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]
        # 转换为PyTorch张量
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)
        if self.half:
            img = img.half()
        return img

    def process(self, img):
        """处理单张图像"""
        self.model.eval()
        with torch.no_grad():
            output = self.model(img)
        return output

    def tile_process(self, img):
        """分块处理大图像"""
        batch, channel, height, width = img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # 如果图像小于tile_size，直接处理
        if height < self.tile_size or width < self.tile_size:
            return self.process(img)

        # 分块处理
        output = img.new_zeros(output_shape)
        tiles_x = width // self.tile_size
        tiles_y = height // self.tile_size

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size

                # 计算输入tile
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # 添加padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # 提取tile
                input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # 处理tile
                output_tile = self.process(input_tile)

                # 计算输出位置
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # 计算在output_tile中的位置
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + (input_end_x - input_start_x) * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + (input_end_y - input_start_y) * self.scale

                # 复制到输出
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                    output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]

        return output

    def post_process(self, output):
        """后处理"""
        # 转换为numpy数组并限制范围
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output, (1, 2, 0))
        # 转换为8位图像
        output = (output * 255.0).round().astype(np.uint8)
        return output

    def enhance(self, img_path, output_path):
        """增强单张图像"""
        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")

        # 预处理
        img_tensor = self.pre_process(img)

        # 推理(分块或整体处理)
        if self.tile_size > 0:
            output = self.tile_process(img_tensor)
        else:
            output = self.process(img_tensor)

        # 后处理
        output_img = self.post_process(output)

        # 保存
        cv2.imwrite(output_path, output_img)
        print(f"增强完成: {output_path}")

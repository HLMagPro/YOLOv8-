import torch
from torch import nn
from torch.nn import init


class ColorAttention(nn.Module):

    def __init__(self, channel):
        super().__init__()
        # 颜色空间转换
        self.color_transform = nn.Sequential(
            nn.Conv2d(channel, channel // 2, 1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(),
            nn.Conv2d(channel // 2, 3, 1)  # 压缩到RGB三通道
        )

        # 颜色特征提取
        self.color_feature = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 转换到颜色空间
        color_map = self.color_transform(x)
        # 提取颜色特征
        color_attention = self.color_feature(color_map)
        return self.sigmoid(color_attention)


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 增强版颜色卷积
        self.color_conv = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel)
        )

        # HSV颜色特征处理分支
        self.hsv_branch = nn.Sequential(
            nn.Conv2d(channel, channel // 4, 1),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(),
            nn.Conv2d(channel // 4, channel, 1)
        )

        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        color_feat = self.color_conv(x)
        hsv_feat = self.hsv_branch(x)

        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        color_out = self.se(self.maxpool(color_feat))
        hsv_out = self.se(self.maxpool(hsv_feat))

        output = self.sigmoid(max_out + avg_out + color_out + hsv_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=9):
        super().__init__()
        self.conv = nn.Conv2d(4, 1, kernel_size=kernel_size, padding=kernel_size // 2)

        # 颜色空间特征提取
        self.color_spatial = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 基础特征
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        color_feat = torch.mean(x, dim=1, keepdim=True)

        # 颜色空间特征
        b, c, h, w = x.shape
        color_spatial = self.color_spatial(x[:, :3] if c > 3 else x)

        result = torch.cat([max_result, avg_result, color_feat, color_spatial], 1)
        output = self.conv(result)
        return self.sigmoid(output)


class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=8, kernel_size=9):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.color_att = ColorAttention(channel)

        # 颜色增强模块
        self.color_enhance = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, groups=channel),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            # 添加深度可分离卷积增强颜色感知
            nn.Conv2d(channel, channel, 3, padding=1, groups=channel),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

        # 可学习参数
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # 颜色增强
        color_enhanced = self.color_enhance(x)

        # 三种注意力机制
        ca_output = x * (self.ca(color_enhanced) / self.temperature)
        sa_output = ca_output * (self.sa(ca_output) / self.temperature)
        color_output = sa_output * self.color_att(color_enhanced)

        # 残差连接和特征融合
        output = (self.alpha * sa_output +
                  self.beta * color_enhanced +
                  self.gamma * color_output +
                  x)  # 残差连接

        return output


if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    kernel_size = input.shape[2]
    cbam = CBAMBlock(channel=512, reduction=8, kernel_size=kernel_size)
    cbam.init_weights()
    output = cbam(input)
    print(output.shape)

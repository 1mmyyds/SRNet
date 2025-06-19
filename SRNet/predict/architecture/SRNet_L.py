import torch
import torch.nn as nn

class L_block(nn.Module):
    def __init__(self, dim):
        super(L_block, self).__init__()
        self.conv1 = nn.Conv2d(dim, 128, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(32, 16, 1, 1, 0, bias=False)
        self.conv_fusion = nn.Conv2d(96, 32, 1, 1, 0, bias=False)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        # 注意力机制
        self.channel_attention = ChannelAttention(in_channels=32)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        # 残差连接
        self.res = nn.Conv2d(dim, 32, 1, 1, 0, bias=False)  # 将输入通道数转换为32

    def forward(self, x):
        """
        x: [b, c, h, w]
        return out: [b, c + 32, h, w]
        """
        feat = self.relu(self.conv1(x))  # [b, 128, h, w]
        feat1 = self.relu(self.conv2(feat))  # [b, 32, h, w]
        feat1_1 = self.relu(self.conv3(feat1))  # [b, 16, h, w]
        feat2 = self.relu(self.conv2(feat))  # [b, 32, h, w]
        feat2_2 = self.relu(self.conv3(feat2))  # [b, 16, h, w]
        feat_fusion = torch.cat([feat1, feat1_1, feat2, feat2_2], dim=1)  # [b, 96, h, w]
        feat_fusion = self.relu(self.conv_fusion(feat_fusion))  # [b, 32, h, w]

        # 应用注意力机制
        feat_fusion = self.channel_attention(feat_fusion)
        feat_fusion = self.spatial_attention(feat_fusion)

        residual = self.res(x)  # [b, 32, h, w]
        feat_fusion = feat_fusion + residual  # [b, 32, h, w]

        out = torch.cat([x, feat_fusion], dim=1)  # [b, c + 32, h, w]
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        通道注意力模块
        :param in_channels: 输入特征图的通道数
        :param reduction: 通道数缩减比例
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()

        # 全局平均池化
        avg_pool = self.avg_pool(x).view(batch, channels)
        avg_out = self.fc(avg_pool)

        # 全局最大池化
        max_pool = self.max_pool(x).view(batch, channels)
        max_out = self.fc(max_pool)

        # 通道注意力权重
        out = avg_out + max_out
        out = self.sigmoid(out).view(batch, channels, 1, 1)

        return x * out.expand_as(x)




class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """
        空间注意力模块
        :param kernel_size: 卷积核大小
        """
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算通道维度的平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [batch, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [batch, 1, H, W]
        concat = torch.cat([avg_out, max_out], dim=1)  # [batch, 2, H, W]

        # 空间注意力权重
        out = self.conv(concat)
        out = self.sigmoid(out)  # [batch, 1, H, W]

        return x * out.expand_as(x)


class shallow_feature_extraction_I(nn.Module):
    def __init__(self, dim):
        super(shallow_feature_extraction_I, self).__init__()

        self.conv1 = nn.Conv2d(dim, 32, 3, 1, 1, bias=False)

        # 注意力机制
        self.channel_attention1 = ChannelAttention(in_channels=32)
        self.spatial_attention1 = SpatialAttention(kernel_size=7)

        self.conv2 = nn.Conv2d(dim, 32, 3, 1, 1, bias=False)

        # 注意力机制
        self.channel_attention2 = ChannelAttention(in_channels=32)
        self.spatial_attention2 = SpatialAttention(kernel_size=7)

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: [b, c, h, w]
        return out: [b, c', h, w]
        """
        feat1 = self.relu(self.conv1(x))  # [b, 32, h, w]
        feat1 = self.channel_attention1(feat1)
        feat1 = self.spatial_attention1(feat1)

        feat2 = self.relu(self.conv2(x))  # [b, 32, h, w]
        feat2 = self.channel_attention2(feat2)
        feat2 = self.spatial_attention2(feat2)

        return feat1, feat2

class shallow_feature_extraction_II(nn.Module):
    def __init__(self):
        super(shallow_feature_extraction_II, self).__init__()

        self.conv1 = nn.Conv2d(32, 32, 1, 1, 0, bias=False)

         # 注意力机制
        self.channel_attention1 = ChannelAttention(in_channels=32)
        self.spatial_attention1 = SpatialAttention(kernel_size=7)

        self.conv2 = nn.Conv2d(32, 32, 1, 1, 0, bias=False)

        # 注意力机制
        self.channel_attention2 = ChannelAttention(in_channels=32)
        self.spatial_attention2 = SpatialAttention(kernel_size=7)

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):

        feat1 = self.relu(self.conv1(x1))  # [b, 32, h, w]
        feat1 = self.channel_attention1(feat1)
        feat1 = self.spatial_attention1(feat1)

        feat2 = self.relu(self.conv2(x2))  # [b, 32, h, w]
        feat2 = self.channel_attention2(feat2)
        feat2 = self.spatial_attention2(feat2)


        return feat1, feat2


class SRNet_L(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, num_blocks=30):
        super(SRNet_L, self).__init__()

        self.shallow_feature_extraction_I=shallow_feature_extraction_I(dim=in_channels)
        self.shallow_feature_extraction_II = shallow_feature_extraction_II()
        L_blocks = [L_block(dim=128 + 32 * i) for i in range(num_blocks)]
        self.L_blocks = nn.Sequential(*L_blocks)
        self.conv_out = nn.Conv2d(128 + 32 * num_blocks, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):

        feat1, feat2 = self.shallow_feature_extraction_I(x)  # [b, 32, h, w]
        feat3, feat4 = self.shallow_feature_extraction_II(feat1,feat2)
        feat_fusion = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        out = self.L_blocks(feat_fusion)
        out = self.conv_out(out)
        return out
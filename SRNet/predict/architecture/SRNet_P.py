import torch.nn as nn
import torch
from einops import rearrange
import torch.nn.functional as F

class Dual_stat_Attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Dual_stat_Attention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # x: 输入特征图, 形状为 (batch_size, in_channels, height, width)
        # 计算通道的平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        # 将平均值和最大值进行拼接
        combined = torch.cat([avg_out, max_out], dim=1)  # 形状为 (batch_size, 2, height, width)
        # 通过卷积生成空间注意力图
        attention_map = self.conv1(combined)
        # 使用 Sigmoid 函数归一化到 [0, 1]
        attention_map = torch.sigmoid(attention_map)
        # 通过注意力图调整输入特征图
        out = x * attention_map  # 广播相乘
        return out

class PreNorm(nn.Module):
    def __init__(self, dim, item):
        super().__init__()
        self.item = item
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.item(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class Multi_Conv(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.list = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        in: [b,h,w,c]
        out: [b,h,w,c]
        """
        out = self.list(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)



class MSA(nn.Module):
    def __init__(self, feature_dim, head_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = head_dim
        self.query_proj = nn.Linear(feature_dim, head_dim * num_heads, bias=False)
        self.key_proj = nn.Linear(feature_dim, head_dim * num_heads, bias=False)
        self.value_proj = nn.Linear(feature_dim, head_dim * num_heads, bias=False)
        self.attention_scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.output_proj = nn.Linear(head_dim * num_heads, feature_dim, bias=True)
        self.position_emb = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1, bias=False, groups=feature_dim),
            GELU(),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1, bias=False, groups=feature_dim), )
        self.dim = feature_dim

    def forward(self, input_tensor):
        """
        input_tensor: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = input_tensor.shape
        x = input_tensor.reshape(b,h*w,c)
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (query, key, value))
        v = v
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attention_weights = (k @ q.transpose(-2, -1))
        attention_weights = attention_weights * self.attention_scale
        attention_weights = attention_weights.softmax(dim=-1)
        x = attention_weights @ v
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        content_output = self.output_proj(x).view(b, h, w, c)
        positional_output = self.position_emb(value.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = content_output + positional_output

        return out


class LGF(nn.Module):
    def __init__(self,dim, dim_head, heads, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MSA(feature_dim=dim, head_dim=dim_head, num_heads=heads),
                PreNorm(dim, Multi_Conv(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (MSA, Multi_Conv) in self.blocks:
            x = MSA(x) + x
            x = Multi_Conv(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class SRNet_P(nn.Module):
    def __init__(self, in_dim=32, out_dim=32, dim=32, depths=2, num_blocks=[2, 4, 4]):
        super(SRNet_P, self).__init__()
        self.dim = dim
        self.depths = depths

        self.DSA = Dual_stat_Attention()

        # Input conv
        self.conv_in = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder = nn.ModuleList([])
        dim_stage = dim
        for i in range(depths):
            self.encoder.append(nn.ModuleList([
                LGF(dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = LGF(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder = nn.ModuleList([])
        for i in range(depths):
            self.decoder.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                LGF(dim=dim_stage // 2, num_blocks=num_blocks[depths - 1 - i], dim_head=dim,heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # Output conv
        self.conv_out = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0, std=0.02, a=-2, b=2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):

            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # in
        fea = self.conv_in(x)

        # Encoder
        fea_encoder = []
        for (LGF, DownSample) in self.encoder:
            fea = LGF(fea)
            fea_encoder.append(fea)
            fea = DownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder

        for i, (UpSample, Fusion, LGF) in enumerate(self.decoder):
            fea = UpSample(fea)
            fea = Fusion(torch.cat([fea, self.DSA(fea_encoder[self.depths - 1 - i])], dim=1))
            fea = LGF(fea)

        # out
        out = self.conv_out(fea) + x

        return out

class generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, n_feat=32, stage=3):
        super(generator, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        cores = [SRNet_P(dim=32, depths=2, num_blocks=[2,4,4]) for _ in range(stage)]
        self.core = nn.Sequential(*cores)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        x = self.conv_in(x)
        h = self.core(x)
        h = self.conv_out(h)
        h += x
        return h

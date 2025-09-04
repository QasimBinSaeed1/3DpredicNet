import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

# MixVision Transformer Encoder
class MixVisionTransformer(nn.Module):
    def __init__(self, embed_dims, num_heads, mlp_ratios, qkv_bias, drop_rate, norm_layer, depths):
        super().__init__()
        self.stages = nn.ModuleList()
        self.embed_dims = embed_dims

        for i in range(len(embed_dims)):
            stage = EncoderStage(
                embed_dim=embed_dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                norm_layer=norm_layer
            )
            self.stages.append(stage)

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

class EncoderStage(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, drop_rate, norm_layer):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, norm_layer)
            for _ in range(depth)
        ])
        self.patch_embed = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.patch_embed(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, norm_layer):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, bias=qkv_bias, batch_first=True)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), drop_rate)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        print(f"Input shape: {x.shape}")
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop_rate)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# SegFormer Decoder
class SegFormerDecoder(nn.Module):
    def __init__(self, embed_dims, decoder_dim, num_classes):
        super().__init__()
        self.linear_fuses = nn.ModuleList([
            nn.Linear(embed_dim, decoder_dim) for embed_dim in embed_dims
        ])
        self.linear_cls = nn.Linear(decoder_dim, num_classes)

    def forward(self, features):
        fused_features = []
        for idx, feature in enumerate(features):
            feature = feature.flatten(2).permute(0, 2, 1)  # B, H*W, C
            fused_features.append(self.linear_fuses[idx](feature))

        fusion = sum(fused_features) / len(fused_features)
        logits = self.linear_cls(fusion)
        # Reshape and upsample to 400x400
        logits = logits.permute(0, 2, 1)  # B, C, H*W
        logits = logits.reshape(logits.size(0), logits.size(1), 20, 20)  # Assume 20x20 grid
        logits = F.interpolate(logits, size=(400, 400), mode="bilinear", align_corners=False)  # Upsample

        return logits

# Full SegFormer Model
class SegFormer(nn.Module):
    def __init__(self, embed_dims = [64, 128, 320, 512], num_heads = [1, 2, 5, 8], mlp_ratios =[4, 4, 4, 4], qkv_bias = True, drop_rate = 0.1, norm_layer = nn.LayerNorm, depths = [3, 4, 6, 3], decoder_dim = 256, num_classes = 2):
        self.n_channels=3
        self.bilinear=True
        self.n_classes=num_classes
        super().__init__()
        self.encoder = MixVisionTransformer(embed_dims, num_heads, mlp_ratios, qkv_bias, drop_rate, norm_layer, depths)
        self.decoder = SegFormerDecoder(embed_dims, decoder_dim, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.decoder(features)
        return logits




if __name__ == "__main__":
    # Example Configuration

    net = SegFormer()
    test_input = torch.randn(4, 3, 400, 400)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1e6))
    output = net(test_input)
    print(f"Output shape: {output.shape}")

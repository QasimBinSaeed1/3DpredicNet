import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(x)

class C3k2(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c1, c_, 3, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.shortcut = shortcut

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        out = torch.cat((x1, x2), dim=1)
        out = self.cv3(out)
        if self.shortcut:
            out = out + x
        return out

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))

class C2PSA(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        return self.cv3(torch.cat((x1, x2), dim=1))

class YOLO11mDepth(nn.Module):
    def __init__(self, in_channels=3, nc=1):
        super().__init__()
        self.n_channels = in_channels  # Input channels (3 for RGB)
        self.n_classes = nc  # Output channels (1 for depth map)
        
        # Backbone
        self.backbone = nn.ModuleList([
            # 0: P1/2
            Conv(in_channels, 64, 3, 2),  # 400x400
            # 1: P2/4
            Conv(64, 128, 3, 2),  # 200x200
            # 2
            C3k2(128, 256, False, 0.25),
            # 3: P3/8
            Conv(256, 256, 3, 2),  # 100x100
            # 4
            C3k2(256, 256, False, 0.25),  # Output 256 channels for P3
            # 5: P4/16
            Conv(256, 512, 3, 2),  # 50x50
            # 6
            C3k2(512, 512, True),
            # 7: P5/32
            Conv(512, 1024, 3, 2),  # 25x25
            # 8
            C3k2(1024, 1024, True),
            # 9
            SPPF(1024, 1024, 5),
            # 10
            C2PSA(1024, 1024)
        ])

        # Head
        self.head = nn.ModuleList([
            # 0 (YAML 11): Upsample to 50x50
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # 1 (YAML 12): Concat with P4 (index 6)
            nn.Identity(),  # Placeholder for concat
            # 2 (YAML 13)
            C3k2(1536, 512, False),  # 1024 + 512 = 1536
            # 3 (YAML 14): Upsample to 100x100
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # 4 (YAML 15): Concat with P3 (index 4)
            nn.Identity(),
            # 5 (YAML 16)
            C3k2(768, 256, False),  # 512 + 256 = 768
            # 6 (YAML 17): Upsample to 200x200
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # 7 (YAML 18): Concat with P2 (index 1)
            nn.Identity(),
            # 8 (YAML 19)
            C3k2(384, 128, False),  # 256 + 128 = 384
            # 9 (YAML 20): Upsample to 800x800
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            # 10 (YAML 21)
            Conv(128, 64, 3, 1),
            # 11 (YAML 22)
            Conv(64, nc, 3, 1),
            # 12 (YAML 23)
            nn.Sigmoid()
        ])

    def forward(self, x):
        # Backbone
        outputs = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [1, 4, 6]:  # Save P2, P3, P4 for concatenation
                outputs.append(x)

        # Head
        concat_indices = {1: (2, 1536), 4: (1, 768), 7: (0, 384)}  # (output_idx, expected_channels)
        for i, layer in enumerate(self.head):
            if i in concat_indices:
                output_idx, expected_channels = concat_indices[i]
                x = torch.cat((x, outputs[output_idx]), dim=1)
            else:
                x = layer(x)

        return x

if __name__ == "__main__":
    # Note: When using with training code, set --scale 1.0 to ensure 800x800 input
    model = YOLO11mDepth(in_channels=3, nc=1)
    x = torch.randn(1, 3, 800, 800)
    output = model(x)
    print(f"Final output shape: {output.shape}")  # Should be [1, 1, 800, 800]
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
        c_ = int(c2 * e)
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

class YOLO11lDepth(nn.Module):
    def __init__(self, in_channels=3, nc=1):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = nc
        
        self.backbone = nn.ModuleList([
            Conv(in_channels, 64, 3, 2),
            Conv(64, 128, 3, 2),
            C3k2(128, 256, False, 0.25),
            Conv(256, 256, 3, 2),
            C3k2(256, 512, False, 0.25),
            Conv(512, 512, 3, 2),
            C3k2(512, 512, True),
            Conv(512, 1024, 3, 2),
            C3k2(1024, 1024, True),
            SPPF(1024, 1024, 5),
            C2PSA(1024, 1024)
        ])

        self.head = nn.ModuleList([
            nn.Upsample(size=(50, 50), mode='bilinear', align_corners=False),
            nn.Identity(),
            C3k2(1536, 512, False),
            nn.Upsample(size=(100, 100), mode='bilinear', align_corners=False),
            nn.Identity(),
            C3k2(1024, 256, False),
            Conv(256, 256, 3, 2),
            nn.Identity(),
            C3k2(768, 512, False),
            Conv(512, 512, 3, 2),
            nn.Identity(),
            C3k2(1536, 512, False),
            nn.Upsample(size=(50, 50), mode='bilinear', align_corners=False),
            nn.Identity(),
            C3k2(1024, 256, False),
            nn.Upsample(size=(100, 100), mode='bilinear', align_corners=False),
            nn.Identity(),
            C3k2(512, 128, False),
            nn.Upsample(size=(800, 800), mode='bilinear', align_corners=False),
            Conv(128, 64, 3, 1),
            Conv(64, nc, 3, 1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 6, 10]:
                outputs.append(x)

        head_outputs = []
        concat_indices = {
            1: (1, 1536, 'backbone'),
            4: (0, 1024, 'backbone'),
            7: (2, 768, 'head'),
            10: (2, 1536, 'backbone'),
            13: (8, 1024, 'head'),
            16: (5, 512, 'head')
        }
        for i, layer in enumerate(self.head):
            x = layer(x)
            head_outputs.append(x)
            if i in concat_indices:
                output_idx, expected_channels, source = concat_indices[i]
                if source == 'backbone':
                    concat_tensor = outputs[output_idx]
                else:
                    concat_tensor = head_outputs[output_idx]
                x = torch.cat((x, concat_tensor), dim=1)
                assert x.shape[1] == expected_channels, f"Expected {expected_channels} channels, got {x.shape[1]}"

        return x

if __name__ == "__main__":
    model = YOLO11lDepth(in_channels=3, nc=1)
    x = torch.randn(1, 3, 800, 800)
    output = model(x)
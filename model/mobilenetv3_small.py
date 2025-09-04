import torch
import torch.nn as nn
#import torch.nn as nn
import torch.nn.functional as F
from geffnet import  tf_mobilenetv3_small_100
from geffnet.efficientnet_builder import InvertedResidual, Conv2dSame, Conv2dSameExport

class ConvBnRelu(nn.Module):
    """Convenience layer combining a Conv2d, BatchNorm2d, and a ReLU activation.

    Original source of this code comes from
    https://github.com/lingtengqiu/Deeperlab-pytorch/blob/master/seg_opr/seg_oprs.py
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 norm_layer=nn.BatchNorm2d):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_planes, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class MobileNetV3(nn.Module):
    def __init__(self, trunk=tf_mobilenetv3_small_100, classes=2, pretrained=False):
        super(MobileNetV3, self).__init__()
        net = trunk(pretrained=pretrained,
                    norm_layer=nn.BatchNorm2d)
        num_classes=2
        self.early = nn.Sequential(net.conv_stem, net.bn1, net.act1)
        self.n_channels = 3
        self.n_classes = classes
        self.bilinear = True

        net.blocks[2][0].conv_dw.stride = (1, 1)
        net.blocks[4][0].conv_dw.stride = (1, 1)

        for block_num in (2, 3, 4, 5):
            for sub_block in range(len(net.blocks[block_num])):
                sb = net.blocks[block_num][sub_block]
                if isinstance(sb, InvertedResidual):
                    m = sb.conv_dw
                else:
                    m = sb.conv
                if block_num < 4:
                    m.dilation = (2, 2)
                    pad = 2
                else:
                    m.dilation = (4, 4)
                    pad = 4
                # Adjust padding if necessary, but NOT for "same" layers
                assert m.kernel_size[0] == m.kernel_size[1]
                if not isinstance(m, Conv2dSame) and not isinstance(m, Conv2dSameExport):
                    pad *= (m.kernel_size[0] - 1) // 2
                    m.padding = (pad, pad)

        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]
        high_level_ch=576
        num_filters=128
        self.aspp_conv1 = nn.Sequential(
            nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )
        self.aspp_conv2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(49, 49), stride=(16, 20)),
            nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
            nn.Sigmoid(),
        )
        aspp_out_ch = num_filters

        s2_ch=16
        s4_ch=64
        self.convs2 = nn.Conv2d(s2_ch, 32, kernel_size=1, bias=False)
        self.convs4 = nn.Conv2d(16, 64, kernel_size=1, bias=False) #
        self.conv_up1 = nn.Conv2d(aspp_out_ch, num_filters, kernel_size=1)
        self.conv_up2 = ConvBnRelu(num_filters + 64, num_filters, kernel_size=1) # convs4 cat
        self.conv_up3 = ConvBnRelu(num_filters + 32, num_filters, kernel_size=1)
        self.last = nn.Conv2d(num_filters, self.n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x1):
        x = self.early(x1) # 2x
        s2 = x
        x = self.block0(x) # 4x
        s4 = x
        x = self.block1(x) # 8x
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        final = self.block5(x)
        aspp = self.aspp_conv1(final) * F.interpolate(
            self.aspp_conv2(final),
            final.shape[2:],
            mode='bilinear',
            align_corners=True
        )#125 64 64
        #F.interpolate(, size=final.shape[2:])
        y = self.conv_up1(aspp) #128 64 64
        y = F.interpolate(y, size=s4.shape[2:], mode='bilinear', align_corners=False) #128 128 128
        s4= self.convs4(s4)
        y = torch.cat([y, s4], 1)
        y = self.conv_up2(y)
        y = F.interpolate(y, size=s2.shape[2:], mode='bilinear', align_corners=False)

        y = torch.cat([y, self.convs2(s2)], 1)
        y = self.conv_up3(y)
        y = self.last(y)
        y = F.interpolate(y, size=x1.shape[2:], mode='bilinear', align_corners=False)
        # return self.sigmoid(y)
        return self.softmax(y)

    @classmethod
    def load(cls,weights_path):
        #print(f"Loading UNet from path `{weights_path}`")
        model = cls()
        model.load_state_dict(torch.load(weights_path))

        return model

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)
        #print(f"Saved model on path: {save_path}")


def mobilenetv3(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    #if pretrained:
    #    state_dict = torch.load('mobilenetv3_small_67.4.pth.tar') #model dict
    #    model.load_state_dict(state_dict, strict=True)

        # raise NotImplementedError
    return model


if __name__ == '__main__':
    net = MobileNetV3()
    #print('mobilenetv3:\n', net)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    input_size=(1, 3,1024, 512)
    # pip install --upgrade git+https://github.com/kuan-wang/pytorch-OpCounter.git
    import datetime
    net=net.cuda()
    
    x = torch.randn(input_size)
    x=x.cuda()
    a = datetime.datetime.now()
    
    #net.eval()
    for i in range(100):
        out = net(x)
        #print(out.shape)

    b = datetime.datetime.now()
    c = b - a
    
    print(c)
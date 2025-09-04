# lines for winodows OS
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
### remove above 2 lines for linux OS
import torch
import torch.nn as nn
import torch.nn.functional as F
# from inplace_abn import InPlaceABN, InPlaceABNSync0


# %% defining depthwise separable convolutions

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, n_stride, n_padding):
        super(depthwise_separable_conv, self).__init__()
        kernels_per_layer = 1
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=n_padding, groups=nin,
                                   stride=n_stride)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)
        self.bn = nn.BatchNorm2d(int(nout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.relu(out)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class depthwise_separable_conv_sig(nn.Module):
    def __init__(self, nin, nout, n_stride, n_padding):
        super(depthwise_separable_conv_sig, self).__init__()
        kernels_per_layer = 1
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=n_padding, groups=nin,
                                   stride=n_stride)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)
        self.bn = nn.BatchNorm2d(int(nout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.relu(out)
        out = self.pointwise(out)
        out = self.bn(out)
        out = torch.sigmoid(out)
        return out


class CCMSubBlock(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        if d == 1:
            self.conv =nn.Sequential(
                nn.Conv2d(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False,
                          dilation=d),
                nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=False)
            )
        else:
            combine_kernel = 2 * d - 1

            self.conv = nn.Sequential(
                nn.Conv2d(nIn, nIn, kernel_size=(combine_kernel, 1), stride=stride, padding=(padding - 1, 0),
                          bias=False),
                nn.BatchNorm2d(nIn),
                nn.PReLU(nIn),
                nn.Conv2d(nIn, nIn, kernel_size=(1, combine_kernel), stride=stride, padding=(0, padding - 1),
                          bias=False),
                nn.BatchNorm2d(nIn),
                nn.Conv2d(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                          dilation=d),
                nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=False))

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

# RCCM  Residual concentrated comprehensive convolution modules (CCM)
# In a similar manner, the depth of channel of RCCM channel is also reduced by one third
# before feding it to the SB-1, SB-2, and SB-3.
#  concentrated comprehensive convolution modules (CCM)
#  In RCCM the input is added to the output of the three sub blocks SB1, SB2, and SB3
# The residual use the concepts of the residual network.
class RCCMModule(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, nIn, nOut, down = True, ratio=[2,4,8]):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        self.down = down
        n = int(nOut // 3)
        n1 = nOut - 3 * n
        #  adding asymmetric convolution layer
        if down:
            self.c1 = nn.Conv2d(nIn, n, 1, 2)
        else:
            self.c1 = nn.Conv2d(nIn, n, 1, 1)
        # SB-1 Sublock 1
        self.d1 = CCMSubBlock(n, n + n1, 3, 1, ratio[0])
        
        self.d1 = nn.Conv2d(n, n+n1, kernel_size=3, padding=1)
        # # SB-2 Sublock 2
        self.d2 = CCMSubBlock(n, n, 3, 1, ratio[1])
        # # SB-2 Sublock 2
        self.d3 = CCMSubBlock(n, n, 3, 1, ratio[2])

        self.bn = nn.Sequential(
            nn.BatchNorm2d(nOut),
            nn.ReLU(inplace=True)
        )


        self.add = nIn == nOut

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)

        combine = torch.cat([d1, d2, d3], 1)

        if self.add and not self.down:
            # print(f"input: {input.shape} combine: {combine.shape}")
            combine = input + combine
        output = self.bn(combine)
        return output


# RCCM  Residual concentrated comprehensive convolution modules (CCM)
# In a similar manner, the depth of channel of RCCM channel is also reduced by one third
# before feding it to the SB-1, SB-2, and SB-3.
#  concentrated comprehensive convolution modules (CCM)
#  In RCCM the input is added to the output of the three sub blocks SB1, SB2, and SB3
# The residual use the concepts of the residual network.
# %% Attention based on Dual attention module ( this performs better) and building the model
class Attention(nn.Module):
    def __init__(self, n_input_channels, device=None):
        super(Attention, self).__init__()

        self.fc = nn.Sequential(nn.Linear(n_input_channels, n_input_channels // 4, 1),
                                nn.ReLU(),
                                nn.Linear(n_input_channels // 4, n_input_channels, 1))

        self.dilated_conv1 = nn.Sequential(
            nn.Conv2d(n_input_channels, n_input_channels, kernel_size=5, stride=1, padding='same', dilation=4,
                      groups=n_input_channels),
            nn.Conv2d(n_input_channels, n_input_channels // 2, kernel_size=1))
        self.dilated_conv2 = nn.Sequential(
            nn.Conv2d(n_input_channels, n_input_channels, kernel_size=5, stride=1, padding='same', dilation=8,
                      groups=n_input_channels),
            nn.Conv2d(n_input_channels, n_input_channels // 2, kernel_size=1))
        self.dilated_conv3 = nn.Sequential(
            nn.Conv2d(n_input_channels, n_input_channels, kernel_size=5, stride=1, padding='same', dilation=12,
                      groups=n_input_channels),
            nn.Conv2d(n_input_channels, n_input_channels // 2, kernel_size=1))

        self.conv1by1 = nn.Conv2d(in_channels=n_input_channels * 3 // 2, out_channels=n_input_channels, kernel_size=1,
                                  stride=1, padding='same')

    def forward(self, input):
        dimention = input.size()
        avg_pool = F.adaptive_avg_pool2d(input, (1, 1))
        avg_pool = torch.flatten(avg_pool, 1)

        max_pool = F.adaptive_max_pool2d(input, (1, 1))
        max_pool = torch.flatten(max_pool, 1)

        max_pool = self.fc(max_pool)
        avg_pool = self.fc(avg_pool)

        channel_attention = torch.sigmoid(avg_pool + max_pool)
        channel_attention = input * channel_attention.view(dimention[0], dimention[1], 1, 1)

        dilated_conv1 = self.dilated_conv1(channel_attention)
        dilated_conv2 = self.dilated_conv2(channel_attention)
        dilated_conv3 = self.dilated_conv3(channel_attention)

        spatial_attention = torch.cat((dilated_conv1, dilated_conv2, dilated_conv3), 1)
        spatial_attention = torch.sigmoid(self.conv1by1(spatial_attention))

        Dual_attention = channel_attention * spatial_attention
        return Dual_attention

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim, device='cuda'):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.device = device

    def INF(self, B, H, W):
        tinf = torch.tensor(float("inf")).to(device=self.device)
        return -torch.diag(tinf.repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        # energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x

class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes=None):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 8
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        print("out_channels", out_channels)
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))

        # self.convb2 = nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels+inter_channels, self.out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.Dropout2d(0.1),
            # nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        self.gap = Global_Avg_Pooling(self.inter_channels, self.inter_channels)

    def forward(self, x, recurrence=1):
        output = self.conva(x) # [B, C/4, H/2, W/2]
        output_gap = self.gap(output)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output) # [B, C/4, H/2, W/2]
        output = self.bottleneck(torch.cat([x, output, output_gap], 1))
        return output

class Global_Avg_Pooling(nn.Module):
    def __init__(self, in_features, out_features):
        super(Global_Avg_Pooling, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.squeeze = nn.Sequential(
            nn.Conv2d(self.in_features, self.out_features, 1, 1),
            nn.BatchNorm2d(self.out_features),
            nn.ReLU()
        )
        self.vectorize = nn.Sequential(
            nn.Linear(self.out_features, self.out_features//2, bias=True),
            nn.ReLU(),
            nn.Linear(self.out_features//2, self.out_features, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        feature_map = self.squeeze(x)
        feature_vector = self.vectorize(feature_map.mean(dim=(-2, -1)))
        return x * feature_vector.reshape(-1, self.out_features, 1, 1)



class NET(nn.Module):
    def __init__(self ,features=[16, 32, 64, 128 ,256], classes=2, device='cuda', use_criss_cross=True):
        super(NET, self).__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3, 3*2, kernel_size=7, padding=3, groups=3, stride=2),
            # add a 3 x 3 convolution layer with 16 output channels
            nn.Conv2d(3*2, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, features[0], kernel_size=1),
            nn.ReLU())
        self.device = device
        self.dropout = nn.Dropout(0.2)
        self.classes = classes
        self.dualAttention = Attention
        self.CCAttention = CrissCrossAttention

        self.n_channels = 3
        self.n_classes = classes
        self.bilinear = True

        self.RCCABlock1 = RCCAModule(features[4], features[4])
        self.RCCABlock2 = RCCAModule(features[4], features[4])

        if not use_criss_cross:
            self.CCAttention = Attention

        self.down1 = RCCMModule(features[0], features[1], down=False)
        self.down2 = RCCMModule(features[1] + features[0], features[2], down=True)
        self.down3 = RCCMModule(features[2], features[2], down=False)
        self.down4 = depthwise_separable_conv(features[2] * 2, features[3], 2, 1)
        self.down5 = depthwise_separable_conv(features[3], features[3], 1, 1)
        self.down6 = depthwise_separable_conv(features[3] * 2, features[4], 2, 1)
        self.mid1 = depthwise_separable_conv(features[4], features[4],  1, 1)
        self.Attention1 = self.dualAttention(features[4], device=self.device)
        self.Conv1 = nn.Conv2d(in_channels=features[4] * 2, out_channels=features[4], kernel_size=1, stride=1,
                               padding='same')
        self.mid2 = depthwise_separable_conv(features[4], features[4], 1, 1)

        self.mid1_1 = depthwise_separable_conv(features[4], features[4], 1, 1)
        self.Attention1_1 = self.dualAttention(features[4], device=self.device)
        self.Conv1_1 = nn.Conv2d(in_channels=features[4] * 2, out_channels=features[4], kernel_size=1, stride=1,
                                 padding='same')
        self.mid2_1 = depthwise_separable_conv(features[4], features[4], 1, 1)

        self.mid1_2 = depthwise_separable_conv(features[4], features[4], 1, 1)
        self.Attention1_2 = self.dualAttention(features[4], device=self.device)
        self.Conv1_2 = nn.Conv2d(in_channels=features[4] * 2, out_channels=features[4], kernel_size=1, stride=1,
                                 padding='same')
        self.mid2_2 = depthwise_separable_conv(features[4], features[4], 1, 1)

        self.UPsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=None)
        self.UP1 = depthwise_separable_conv(features[4], features[3], 1, 1)
        self.UP2 = depthwise_separable_conv(features[3] * 2, features[3], 1, 1)
        self.Attention2 = self.dualAttention(features[3], device=self.device)
        self.Conv2 = nn.Conv2d(in_channels=features[3] * 2, out_channels=features[3], kernel_size=1, stride=1,
                               padding='same')

        self.UPsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=None)
        self.UP3 = depthwise_separable_conv(features[3], features[2], 1, 1)
        self.UP4 = depthwise_separable_conv(features[2] * 2, features[2], 1, 1)
        self.Attention3 = self.dualAttention(features[2], device=self.device)
        self.Conv3 = nn.Conv2d(in_channels=features[2] * 2, out_channels=features[2], kernel_size=1, stride=1,
                               padding='same')

        self.UPsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=None)
        self.UP5 = depthwise_separable_conv(features[2], features[1], 1, 1)
        self.UP6 = depthwise_separable_conv(features[1] * 2, features[1], 1, 1)
        self.Attention4 = self.dualAttention(features[1], device=self.device)
        self.Conv4 = nn.Conv2d(in_channels=features[1] * 2, out_channels=features[1], kernel_size=1, stride=1,
                               padding='same')

        self.UPsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=None)
        self.UP7 = depthwise_separable_conv(features[1], features[0], 1, 1)
        # self.UP8 = depthwise_separable_conv_sig(features[0], 1, 1, 1)
        # self.UP8 = depthwise_separable_conv_sig(features[0],3,1,1)
        self.UP8_mask = depthwise_separable_conv_sig(features[0],self.classes,1,1)
        # self.UP9 = depthwise_separable_conv_sig(4,1,1,1)

    def forward(self, input):  # H*W
        x0 = self.entry(input)  # H/2*W/2
        x1 = self.down1(x0)  # H/2*W/2
        x2 = torch.cat((x0, x1), 1)
        x2 = self.down2(x2)  # H/4*W/4
        x2 = self.dropout(x2)
        x3 = self.down3(x2)  # H/4*W/4
        x4 = torch.cat((x2, x3), 1)
        x4 = self.down4(x4)  # H/8*W/8
        x4 = self.dropout(x4)
        x5 = self.down5(x4)  # H/8*W/8
        x6 = torch.cat((x4, x5), 1)
        x6 = self.down6(x6)  # H/16*W/16
        x6 = self.dropout(x6)
        mid1 = self.mid1(x6)  # H/16*W/16
        attention1 = self.Attention1(x6)
        conv1 = torch.relu(self.Conv1(torch.cat((attention1, mid1), 1)))
        mid2 = self.mid2(conv1)  # H/16*W/16

        mid1_1 = self.mid1_1(mid2)  # H/16*W/16
        attention1_1 = self.Attention1_1(mid2)
        conv1_1 = torch.relu(self.Conv1_1(torch.cat((attention1_1, mid1_1), 1)))
        mid2_1 = self.mid2_1(conv1_1)  # H/16*W/16

        mid1_2 = self.mid1_2(mid2_1)  # H/16*W/16
        attention1_2 = self.Attention1_2(mid2_1)
        conv1_2 = torch.relu(self.Conv1_2(torch.cat((attention1_2, mid1_2), 1)))
        mid2_2 = self.mid2_2(conv1_2)  # H/16*W/16

        # mid2_2 = self.RCCABlock1(mid2_2) # our cc attention block
        # mid2_2 = self.RCCABlock2(mid2_2)

        upsample1 = self.UPsample1(mid2_2)  # H/8*W/8
        up1 = self.UP1(upsample1)  # H/8*W/8
        cat1 = torch.cat((up1, x5), 1)
        up2 = self.UP2(cat1)  # H/8*W/8
        attention2 = self.Attention2(up1)
        conv2 = torch.relu(self.Conv2(torch.cat((attention2, up2), 1)))

        upsample2 = self.UPsample2(conv2)  # H/4*W/4
        up3 = self.UP3(upsample2)  # H/4*W/4
        cat2 = torch.cat((up3, x3), 1)
        up4 = self.UP4(cat2)  # H/4*W/4
        attention3 = self.Attention3(up3)
        conv3 = torch.relu(self.Conv3(torch.cat((attention3, up4), 1)))

        upsample3 = self.UPsample3(conv3)  # H/2*W/2
        up5 = self.UP5(upsample3)  # H/2*W/2
        cat3 = torch.cat((up5, x1), 1)
        up6 = self.UP6(cat3)  # H/2*W/2
        attention4 = self.Attention4(up5)
        conv4 = torch.relu(self.Conv4(torch.cat((attention4, up6), 1)))

        upsample4 = self.UPsample4(conv4)  # H*W
        up7 = self.UP7(upsample4)  # H*W
        # depth = self.UP8(up7)  # H*W
        mask = self.UP8_mask(up7)  # H*W
        # up9 = self.UP9(up8)       #H*W

        return mask
        print("mask:", mask.shape)
    def use_checkpointing(self):
        self.use_checkpoint = True

if __name__ == '__main__':
    # from torchsummary import summary
    #torch.cuda.empty_cache()
    #summary(mynetwork,(3,400,400), device = device)
    #%%

    nn1 = RCCAModule(128, 64)
    img = torch.randn(1, 128, 400, 400)
    out = nn1(img)
    print(out.shape)
    #%%


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    mynetwork = NET(device=device)
    # mynetwork = RCCAModule(128, 64, 3)
    # print(mynetwork)
    # mynetwork = CCMSubBlock(3, 16, 1)
    # mynetwork = RCCMModule(3, 3, down=False)
    # model params count
    print("parameters: ", sum(p.numel() for p in mynetwork.parameters() if p.requires_grad))
    net = mynetwork.to(device)
    img = torch.randn(16, 3, 64, 64).to(device)
    # depth, mask = net(img)
    depth, _ = net(img)
    print("out: ", depth.shape)
    # print(torch.unique(depth))
    # summary(net,(3,400,400), device = 'cuda')
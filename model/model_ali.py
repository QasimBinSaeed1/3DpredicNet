
import torch
import torch.nn as nn
import torch.nn.functional as F

class depthwise_separable_conv(nn.Module):
 def __init__(self, nin, nout,n_stride,n_padding): 
   super(depthwise_separable_conv, self).__init__() 
   kernels_per_layer =1
   self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=n_padding, groups=nin,stride = n_stride) 
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
 def __init__(self, nin, nout,n_stride,n_padding): 
   super(depthwise_separable_conv_sig, self).__init__() 
   kernels_per_layer =1
   self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=n_padding, groups=nin,stride = n_stride) 
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


#%% Attention based on Atrous Special Pooling Pyramid
class Attention(nn.Module):
    def __init__(self,n_input_channels):
        super(Attention, self).__init__()

        self.fc = nn.Sequential(nn.Linear(n_input_channels, n_input_channels // 4, 1),
                               nn.ReLU(),
                               nn.Linear(n_input_channels // 4, n_input_channels, 1))

        self.PW = nn.Conv2d(in_channels =n_input_channels, out_channels = n_input_channels//2, kernel_size = 1, stride=1, padding='same')
        self.dilated_conv1 = nn.Sequential(nn.Conv2d(n_input_channels,n_input_channels, kernel_size=5, stride=1, padding='same',dilation = 2, groups=n_input_channels),
                                           nn.Conv2d(n_input_channels, n_input_channels//2, kernel_size=1),
                                           nn.ReLU()) 
        self.dilated_conv2 = nn.Sequential(nn.Conv2d(n_input_channels,n_input_channels, kernel_size=5, stride=1, padding='same',dilation = 4, groups=n_input_channels),
                                           nn.Conv2d(n_input_channels, n_input_channels//2, kernel_size=1),
                                           nn.ReLU()) 
        self.dilated_conv3 = nn.Sequential(nn.Conv2d(n_input_channels,n_input_channels, kernel_size=5, stride=1, padding='same',dilation = 8, groups=n_input_channels),
                                           nn.Conv2d(n_input_channels, n_input_channels//2, kernel_size=1),
                                           nn.ReLU()) 
        
        self.conv1by1 = nn.Conv2d(in_channels =n_input_channels*2, out_channels = n_input_channels, kernel_size = 1, stride=1, padding='same')

    def forward(self, input):
          dimention = input.size()
          avg_pool = F.adaptive_avg_pool2d(input, (1, 1))
          avg_pool = torch.flatten(avg_pool, 1)

          max_pool = F.adaptive_max_pool2d(input, (1, 1))
          max_pool = torch.flatten(max_pool, 1)

          max_pool = self.fc(max_pool)
          avg_pool = self.fc(avg_pool)

          channel_attention = torch.sigmoid(avg_pool + max_pool)
          channel_attention = input*channel_attention.view(dimention[0],dimention[1],1,1)

          dilated_conv1 = self.dilated_conv1(input)
          dilated_conv2 = self.dilated_conv2(input)
          dilated_conv3 = self.dilated_conv3(input)
          pw = self.PW(input)
          
          spatial_attention = torch.cat((dilated_conv1, dilated_conv2,dilated_conv3,pw), 1)
          spatial_attention = torch.sigmoid(self.conv1by1(spatial_attention)) 
          
          attention = torch.cat((channel_attention,spatial_attention),1)
          return attention

#%% Attention based on Dual attention module ( this performs better) and building the model
class Attention(nn.Module):
    def __init__(self,n_input_channels):
        super(Attention, self).__init__()

        self.fc = nn.Sequential(nn.Linear(n_input_channels, n_input_channels // 4, 1),
                               nn.ReLU(),
                               nn.Linear(n_input_channels // 4, n_input_channels, 1))

        #self.dilated_conv1 = dilated_separable_conv(n_input_channels,1,n_input_channels//2,4)
        #self.dilated_conv2 = dilated_separable_conv(n_input_channels,1,n_input_channels//2,8)
        #self.dilated_conv3 = dilated_separable_conv(n_input_channels,1,n_input_channels//2,12)
        self.dilated_conv1 = nn.Sequential(nn.Conv2d(n_input_channels,n_input_channels, kernel_size=5, stride=1, padding='same',dilation = 4, groups=n_input_channels),
                                           nn.Conv2d(n_input_channels, n_input_channels//2, kernel_size=1)) 
        self.dilated_conv2 = nn.Sequential(nn.Conv2d(n_input_channels,n_input_channels, kernel_size=5, stride=1, padding='same',dilation = 8, groups=n_input_channels),
                                           nn.Conv2d(n_input_channels, n_input_channels//2, kernel_size=1)) 
        self.dilated_conv3 = nn.Sequential(nn.Conv2d(n_input_channels,n_input_channels, kernel_size=5, stride=1, padding='same',dilation = 12, groups=n_input_channels),
                                           nn.Conv2d(n_input_channels, n_input_channels//2, kernel_size=1)) 
        
        
        self.conv1by1 = nn.Conv2d(in_channels =n_input_channels*3//2, out_channels = n_input_channels, kernel_size = 1, stride=1, padding='same')

    def forward(self, input):
          dimention = input.size()
          avg_pool = F.adaptive_avg_pool2d(input, (1, 1))
          avg_pool = torch.flatten(avg_pool, 1)

          max_pool = F.adaptive_max_pool2d(input, (1, 1))
          max_pool = torch.flatten(max_pool, 1)

          max_pool = self.fc(max_pool)
          avg_pool = self.fc(avg_pool)

          channel_attention = torch.sigmoid(avg_pool + max_pool)
          channel_attention = input*channel_attention.view(dimention[0],dimention[1],1,1)

          dilated_conv1 = self.dilated_conv1(channel_attention)
          dilated_conv2 = self.dilated_conv2(channel_attention)
          dilated_conv3 = self.dilated_conv3(channel_attention)

          spatial_attention = torch.cat((dilated_conv1, dilated_conv2,dilated_conv3), 1)
          spatial_attention = torch.sigmoid(self.conv1by1(spatial_attention)) 

          Dual_attention = channel_attention*spatial_attention
          return Dual_attention



class NET(nn.Module):
    def __init__(self,features=[16, 32, 64, 128,256], device='cuda', classes=3):
        super(NET, self).__init__()
        self.entry = nn.Sequential(
        nn.Conv2d(3, 3*2, kernel_size=7, padding=3, groups=3,stride =2), 
        nn.Conv2d(3*2, features[0], kernel_size=1),
        nn.ReLU()) 

        self.n_channels = 3
        self.n_classes = classes
        self.bilinear = True
        
        self.dropout = nn.Dropout(0.2)
        
        self.down1 = depthwise_separable_conv(features[0],features[1],1,1)
        self.down2 = depthwise_separable_conv(features[1]+features[0],features[2],2,1)
        self.down3 = depthwise_separable_conv(features[2],features[2],1,1)
        self.down4 = depthwise_separable_conv(features[2]*2,features[3],2,1)
        self.down5 = depthwise_separable_conv(features[3],features[3],1,1)
        self.down6 = depthwise_separable_conv(features[3]*2,features[4],2,1)
        self.mid1 = depthwise_separable_conv(features[4],features[4],1,1)
        self.Attention1 = Attention(features[4])
        self.Conv1 =  nn.Conv2d(in_channels =features[4]*2, out_channels = features[4], kernel_size = 1, stride=1, padding='same')
        self.mid2 = depthwise_separable_conv(features[4],features[4],1,1)
        
        self.mid1_1 = depthwise_separable_conv(features[4],features[4],1,1)
        self.Attention1_1 = Attention(features[4])
        self.Conv1_1 =  nn.Conv2d(in_channels =features[4]*2, out_channels = features[4], kernel_size = 1, stride=1, padding='same')
        self.mid2_1 = depthwise_separable_conv(features[4],features[4],1,1)
        
        self.mid1_2 = depthwise_separable_conv(features[4],features[4],1,1)
        self.Attention1_2 = Attention(features[4])
        self.Conv1_2 =  nn.Conv2d(in_channels =features[4]*2, out_channels = features[4], kernel_size = 1, stride=1, padding='same')
        self.mid2_2 = depthwise_separable_conv(features[4],features[4],1,1)

        self.UPsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=None)
        self.UP1 = depthwise_separable_conv(features[4],features[3],1,1)
        self.UP2 = depthwise_separable_conv(features[3]*2,features[3],1,1)
        self.Attention2 = Attention(features[3])
        self.Conv2 =  nn.Conv2d(in_channels =features[3]*2, out_channels = features[3], kernel_size = 1, stride=1, padding='same')
        
        self.UPsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=None)
        self.UP3 = depthwise_separable_conv(features[3],features[2],1,1)
        self.UP4 = depthwise_separable_conv(features[2]*2,features[2],1,1)
        self.Attention3 = Attention(features[2])
        self.Conv3 =  nn.Conv2d(in_channels =features[2]*2, out_channels = features[2], kernel_size = 1, stride=1, padding='same')

        self.UPsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=None)
        self.UP5 = depthwise_separable_conv(features[2],features[1],1,1)
        self.UP6 = depthwise_separable_conv(features[1]*2,features[1],1,1)
        self.Attention4 = Attention(features[1])
        self.Conv4 =  nn.Conv2d(in_channels =features[1]*2, out_channels = features[1], kernel_size = 1, stride=1, padding='same')
        
        self.UPsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=None)
        self.UP7 = depthwise_separable_conv(features[1],features[0],1,1)
        self.UP8 = depthwise_separable_conv_sig(features[0],self.n_classes,1,1)
        #self.UP9 = depthwise_separable_conv_sig(4,1,1,1)

  
    def forward(self, input):           #H*W
        x0 = self.entry(input)          #H/2*W/2
        x1 = self.down1(x0)             #H/2*W/2
        x2 = torch.cat((x0, x1), 1)
        x2 = self.down2(x2)             #H/4*W/4
        x2 = self.dropout(x2)
        x3 = self.down3(x2)             #H/4*W/4
        x4 = torch.cat((x2, x3), 1)
        x4 = self.down4(x4)             #H/8*W/8
        x4 = self.dropout(x4)
        x5 = self.down5(x4)             #H/8*W/8
        x6 = torch.cat((x4, x5), 1)
        x6 = self.down6(x6)             #H/16*W/16
        x6 = self.dropout(x6)
        mid1 = self.mid1(x6)            #H/16*W/16
        attention1 = self.Attention1(x6)
        conv1 = torch.relu(self.Conv1(torch.cat((attention1,mid1),1)))
        mid2 = self.mid2(conv1)                     #H/16*W/16
        
        mid1_1 = self.mid1_1(mid2)                  #H/16*W/16
        attention1_1 = self.Attention1_1(mid2)
        conv1_1 = torch.relu(self.Conv1_1(torch.cat((attention1_1,mid1_1),1)))
        mid2_1 = self.mid2_1(conv1_1)               #H/16*W/16
        
        
        mid1_2 = self.mid1_2(mid2_1)                  #H/16*W/16
        attention1_2 = self.Attention1_2(mid2_1)
        conv1_2 = torch.relu(self.Conv1_2(torch.cat((attention1_2,mid1_2),1)))
        mid2_2 = self.mid2_2(conv1_2)               #H/16*W/16
        
        upsample1 = self.UPsample1(mid2_2)          #H/8*W/8
        up1 = self.UP1(upsample1)                   #H/8*W/8
        cat1 = torch.cat((up1,x5),1)    
        up2 = self.UP2(cat1)                        #H/8*W/8
        attention2 = self.Attention2(up1)
        conv2 = torch.relu(self.Conv2(torch.cat((attention2,up2),1)))
        
        upsample2 = self.UPsample2(conv2)           #H/4*W/4
        up3 = self.UP3(upsample2)                   #H/4*W/4
        cat2 = torch.cat((up3,x3),1)    
        up4 = self.UP4(cat2)                        #H/4*W/4
        attention3 = self.Attention3(up3)
        conv3 = torch.relu(self.Conv3(torch.cat((attention3,up4),1)))
        
        upsample3 = self.UPsample3(conv3)            #H/2*W/2
        up5 = self.UP5(upsample3)                    #H/2*W/2
        cat3 = torch.cat((up5,x1),1)    
        up6 = self.UP6(cat3)                         #H/2*W/2
        attention4 = self.Attention4(up5)
        conv4 = torch.relu(self.Conv4(torch.cat((attention4,up6),1)))
        
        upsample4 = self.UPsample4(conv4)            #H*W
        up7 = self.UP7(upsample4)                    #H*W
        up8 = self.UP8(up7)                          #H*W
        #up9 = self.UP9(up8)       #H*W

        return up8


# class NET(nn.Module):
    # def __init__(self,features=[16, 32, 64, 128,256]):


net = NET().cuda()

batch = torch.rand(1,3,256,256).cuda()
out = net(batch)

print(out.shape)
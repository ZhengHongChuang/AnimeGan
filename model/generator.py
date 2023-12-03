import math
import torch
from torch import nn as nn
from .norm import Norm
def truncated_normal_(weight, mean=0., std=0.1):
    # 生成随机标准正态分布
    size = weight.shape
    tmp = weight.new_empty(size + (4,)).normal_()
    # 保留范围在 (-2, 2) 内的样本
    valid = (tmp < 2) & (tmp > -2)
    # 获取在范围内的索引
    ind = valid.max(-1, keepdim=True)[1]
    # 从 tmp 中选择合法的样本
    weight.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    # 标准化到指定的均值和标准差
    weight.data.mul_(std).add_(mean)
    return weight
class Conv_Norm_ReLU(nn.Module):
    def __init__(self, in_channel,out_channel,kernel_size,stride = 1,padding = 0,bias = False):
        super().__init__()
        self.Conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size,
                              stride,
                              padding,
                              padding_mode= 'reflect',
                              bias = bias)
        self.Norm = Norm()
        self.ReLU = nn.LeakyReLU(0.2)
    def forward(self,x):
        x = self.Conv(x)
        x = self.Norm(x)
        x = self.ReLU(x)
        return x
class Res_Block(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super().__init__()
        self.add_res = (in_channels == out_channels and stride == 1)
        self.pw = Conv_Norm_ReLU(in_channel = in_channels,out_channel=2*in_channels,kernel_size=1,stride=1,padding=0)
        self.dw = nn.Sequential(
            nn.Conv2d(
                2*in_channels,
                2*in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=2*in_channels,
                padding_mode='reflect'
            ),
            Norm(),
            nn.LeakyReLU(0.2)
        )
        self.pw_linear = nn.Sequential(
            nn.Conv2d(in_channels = 2*in_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0,
                      bias = False,
                      padding_mode='reflect'),
            Norm()  
        )
    def forward(self,x):
        out = self.pw(x)
        out = self.dw(out)
        out = self.pw_linear(out)
        if self.add_res:
            out += x
        return out
class G(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.Enconding = nn.Sequential(
            Conv_Norm_ReLU(in_channel,out_channel = 32 ,kernel_size= 7 , stride = 1 , padding = 3),
            Conv_Norm_ReLU(in_channel = 32,out_channel = 64 ,kernel_size = 3,stride = 2, padding = 1),
            Conv_Norm_ReLU(in_channel = 64 ,out_channel = 64 ,kernel_size = 3,stride = 1 ,padding = 1),
            Conv_Norm_ReLU(in_channel = 64,out_channel = 128 ,kernel_size = 3,stride = 2, padding = 1),
            Conv_Norm_ReLU(in_channel = 128 ,out_channel = 128 ,kernel_size = 3,stride = 1 ,padding = 1),
            Conv_Norm_ReLU(in_channel = 128,out_channel = 128 ,kernel_size = 3,stride = 1, padding = 1),
            Res_Block(128,256,1),
            Res_Block(256,256,1),
            Res_Block(256,256,1),
            Res_Block(256,256,1),
            Conv_Norm_ReLU(in_channel = 256,out_channel = 128,kernel_size = 3,padding = 1)
        )
        self.Decoding = nn.Sequential(
            nn.Upsample(scale_factor = 2,mode='bilinear',align_corners=False),
            Conv_Norm_ReLU(in_channel = 128,out_channel = 128,kernel_size = 3,padding = 1),
            Conv_Norm_ReLU(in_channel = 128,out_channel = 128,kernel_size = 3,padding = 1),
            nn.Upsample(scale_factor = 2,mode='bilinear',align_corners=False),
            Conv_Norm_ReLU(in_channel = 128,out_channel = 64,kernel_size = 3,padding = 1),
            Conv_Norm_ReLU(in_channel = 64,out_channel = 64,kernel_size = 3,padding = 1),
            Conv_Norm_ReLU(in_channel = 64,out_channel = 32,kernel_size = 7,padding = 3),
            nn.Conv2d(
                in_channels = 32,
                out_channels = 3,
                kernel_size = 1,
                stride = 1,
                bias = False,
                padding_mode='reflect'),
            nn.Tanh()
        )
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                truncated_normal_(m.weight, mean=0., std=math.sqrt(1.3 * 2.0 / m.in_channels))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.Enconding(x)
        x = self.Decoding(x)
        return x
if __name__ == "__main__": 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import torch

    input_shape = torch.randn(16, 3, 224, 224).to(device)

    model = G(in_channel=3)
    model = model.to(device)
    output = model(input_shape)
    print(output.size())
 

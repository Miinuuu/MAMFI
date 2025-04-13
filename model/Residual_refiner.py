import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mlp(in_c,h_c,out_c,k=1,s=1,p=0):
    return nn.Sequential(
        nn.Conv2d(in_c,h_c,1,1,0),
        nn.PReLU(h_c),
        nn.Conv2d(h_c,out_c,1,1,0)
        )


        
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
        )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
        )
            

class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes,kernel_size=3, stride=2,padding=1):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, kernel_size, stride, padding)
        self.conv2 = conv(out_planes, out_planes, 3, 1, padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
   
class Residual_refiner(nn.Module):
    def __init__(self, c, out=3):
        super(Residual_refiner, self).__init__()
        print("Residual_refiner")
    
        self.up0 = nn.Sequential(
                         mlp(17+4*c,4*c*4, 4*c),
                         Conv2(4*c,4*c,3,1,1),
                         Conv2(4*c,4*c,3,1,1),
                         deconv(4*c, 2*c)
        )
        
        self.up1 =  nn.Sequential(
                        mlp(17+2*c+2*c,2*c*4, 2*c),
                         Conv2(2*c,2*c,3,1,1),
                         Conv2(2*c,2*c,3,1,1),
                        deconv(2*c, c)
        )
        self.up2 = nn.Sequential(
                        mlp(17+c+c,c*4,c),
                         Conv2(c,c,3,1,1),
                         Conv2(c,c,3,1,1),
                        nn.Conv2d(c , out, 3, 1, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((   imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) 
        x = self.up1(torch.cat((x, imgs[-2],wimgs[-2],Cs[-2],flows[-2],masks[-2]), 1))
        x = self.up2(torch.cat((x, imgs[-3],wimgs[-3],Cs[-3],flows[-3],masks[-3]), 1))
        return torch.sigmoid(x)
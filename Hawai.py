import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchsummary import summary

class EncoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(EncoderBlock,self).__init__()
        self.Conv1=nn.Conv2d(in_channels,out_channels,3,1,1)
        self.batchnorm=nn.BatchNorm2d(out_channels)
        self.activation=nn.ReLU(inplace=True)
        self.Conv2=nn.Conv2d(out_channels,out_channels,3,1,1)
        self.batchnorm2=nn.BatchNorm2d(out_channels)
        self.activation2=nn.ReLU(inplace=True)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        x1=self.Conv1(x)
        x2=self.batchnorm(x1)
        x3=self.activation(x2)
        x4=self.Conv2(x3)
        x5=self.batchnorm2(x4)
        x6=self.activation2(x5)
        x7=self.pool(x6)

        return x7
    
class bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(bottleneck,self).__init__()
        self.Conv1=nn.Conv2d(in_channels,out_channels,3,1,1)
        self.activation=nn.ReLU(inplace=True)
        self.Conv2=nn.Conv2d(out_channels,out_channels,3,1,1)
        self.activation2=nn.ReLU(inplace=True)

    def forward(self,x):
        x1=self.Conv1(x)
        x2=self.activation(x1)
        x3=self.Conv2(x2)
        x4=self.activation2(x3)
        return x4
    
class connector(nn.Module):
    def __init__(self,in_channels):
        super(connector,self).__init__()
        self.encoder1=EncoderBlock(in_channels,8)
        self.encoder2=EncoderBlock(8,16)
        self.encoder3=EncoderBlock(16,32)
        self.bottleneck=bottleneck(32,64)

    def forward(self,x):
        x1=self.encoder1(x)
        x2=self.encoder2(x1)
        x3=self.encoder3(x2)
        x4=self.bottleneck(x3)
        return x4
    
if __name__ == '__main__':

    connector=connector(3)
    summary(connector,(3,512,512))

    x=torch.randn(1,64,64,64)
    x=x[0]
    x=x.permute(1,2,0)
    x=x.reshape(64,64,64)
    x=x.reshape(64,64,8,8)
    x=x.permute(0,2,1,3)
    x=x.reshape(512,512)
    print(x.size())
    print(x)











    








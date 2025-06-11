import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self,input_channels,n_features):
        #input_channels can be 1 (grey scale) or 3 (rgb color); the conv with a kernel of n features, results on an image with channels equals to n_features
        super().__init__() #call the constructor (__init__) of nn.Module for the correct implementation of PyTorch
        self.conv_result=nn.Sequential(
            nn.Conv3d(input_channels,n_features,kernel_size=(3,3,3),padding="same",bias=False),#the resulting image will have the same size as the input one
            nn.InstanceNorm3d(n_features),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            
            nn.Conv3d(n_features,n_features,kernel_size=(3,3,3),padding="same",bias=False),
            nn.InstanceNorm3d(n_features),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.3),
        )
    def forward(self,x): #Instruct how the sequence is applied, is called implicitly in the model
        return self.conv_result(x)

class encoder_block(nn.Module): #Consist in two conv of 3x3 kernel and a 2x2 max pooling with stride value of 2
    def __init__(self,input_channels,n_features):
        super().__init__()
        self.conv_result=conv_block(input_channels,n_features) #two conv of 3x3 kernel
        self.max_pool_result=nn.MaxPool3d(kernel_size=(2,2,2), stride=2,ceil_mode=True) #2x2 max pooling with stride value of 2

    def forward(self,x):
        fx=self.conv_result(x) #signal will be use in the attention gate
        ouput=self.max_pool_result(fx) #ouput will be use for the next procedures of the u-net
        return fx,ouput

class decoder_block(nn.Module):
    def __init__(self,input_channels,n_features):
        super().__init__()
        self.up_result=nn.Upsample(scale_factor=2,mode="trilinear",align_corners=True) #The size of the image input is duplicated by two
        self.ag_result=attention_gate(input_channels,n_features) #Give the spacial information with the corresponding weights
        self.conv_result=conv_block(input_channels[0]+input_channels[1],n_features) #Conv of 3x3 kernel

    def forward(self,x,s):
        y=self.ag_result(x,s)
        x_up=self.up_result(x)
        x_out=self.conv_result(torch.cat((x_up,y),dim=1))
        return x_out


class attention_gate(nn.Module):
    def __init__(self,input_channels,n_features):
        super().__init__()
        #Obtain the weight with the same size of features
        self.weight_g=nn.Sequential(
            nn.Conv3d(input_channels[0],n_features,kernel_size=(1,1,1), stride=1, bias=False),
            nn.InstanceNorm3d(n_features)
        )
        self.weight_fx=nn.Sequential(
            nn.Conv3d(input_channels[1],n_features,kernel_size=(1,1,1), stride=2, bias=False),
            nn.InstanceNorm3d(n_features)
        )    
        self.relu=nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.output=nn.Sequential(
            nn.Conv3d(n_features,1,kernel_size=(1,1,1), padding="same", stride=1, bias=False),
            nn.InstanceNorm3d(1),
            nn.Sigmoid() #Sigmoid is used to scale all the values from 0 to 1
        )

    def forward(self,g,fx):
        weight_g=self.weight_g(g)
        weight_fx=self.weight_fx(fx)
        out=self.relu(weight_g+weight_fx)
        out=self.output(out)
        out = F.interpolate(out, size=fx.size()[2:], mode='trilinear')
        return out*fx #Upsampling to the original size of x
    
class attention_unet_seg(nn.Module):
    def __init__(self,input_channels):
        super().__init__()
        self.e1=encoder_block(input_channels,16)
        self.e2=encoder_block(16,32)
        self.e3=encoder_block(32,64)
        self.e4=encoder_block(64,128)
        
        #The bridge
        self.b1=conv_block(128,256)
        #[a,b]: a: features of the gating signal; b: features of the input from encoder
        self.d1=decoder_block([256,128],128)
        self.d2=decoder_block([128,64],64)
        self.d3=decoder_block([64,32],32)
        self.d4=decoder_block([32,16],16)

        self.output=nn.Conv3d(16,1,kernel_size=(1,1,1),padding="same")

    def forward(self,x):
        #Encoder
        fx_1,p1=self.e1(x)
        fx_2,p2=self.e2(p1)
        fx_3,p3=self.e3(p2)
        fx_4,p4=self.e4(p3)
        #Bridge
        b1=self.b1(p4)
        
        #Decoder
        d1=self.d1(b1,fx_4)
        d2=self.d2(d1,fx_3)
        d3=self.d3(d2,fx_2)
        d4=self.d4(d3,fx_1)
        #Final operation
        out=self.output(d4) #Output is a grey scale volume
        return out

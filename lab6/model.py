import torch
import torch.nn as nn
from torchvision import transforms

class Discriminator(nn.Module):
    def __init__(self,img_shape,condition_dim):
        super(Discriminator, self).__init__()
        self.H,self.W,C=img_shape
        self.condition2img=nn.Linear(condition_dim,1*self.H*self.W)
        kernel_size=(3,3)
        channels=[4,128,256,512]
        for i in range(1,len(channels)):
            setattr(self,'conv'+str(i),nn.Sequential(
                nn.Conv2d(channels[i-1],channels[i],kernel_size,stride=(2,2),padding=(1,1)),
                nn.LeakyReLU()
            ))
        #self.dropout=nn.Dropout()
        self.classify1=nn.Sequential(
            nn.Linear(32768,512),
            nn.LeakyReLU()
        )
        self.classify2=nn.Linear(512,2)

    def forward(self,X,c):
        """
        :param X: (batch_size,3,64,64) tensor
        :param c: (batch_size,24) tensor
        :return: (batch_size,2) tensor
        """
        condition=self.condition2img(c).view(-1,1,self.H,self.W)
        out=torch.cat((condition,X),dim=1)  # become(N,4,64,64)
        out=self.conv1(out)  # become(N,128,32,32)
        out=self.conv2(out)  # become(N,256,16,16)
        out=self.conv3(out)  # become(N,512,8,8)
        out=out.view(out.shape[0],-1)  # become(N,32768)
        out=self.classify1(out)  #become(N,512)
        out=self.classify2(out)  #become(N,2)
        return out

class Generator(nn.Module):
    def __init__(self,z_dim,condition_dim):
        super(Generator,self).__init__()
        self.feature_map_size=8
        self.condition2img=nn.Linear(condition_dim,1*self.feature_map_size*self.feature_map_size)
        self.z2img=nn.Linear(z_dim,512*self.feature_map_size*self.feature_map_size)
        kernel_size=(4,4)
        channels=[513,256,128,128]
        for i in range(1,len(channels)):
            setattr(self,'convT'+str(i),nn.Sequential(
                nn.ConvTranspose2d(channels[i-1],channels[i],kernel_size,stride=(2,2),padding=(1,1)),
                nn.LeakyReLU()
            ))
        self.conv=nn.Sequential(
            nn.Conv2d(128,3,kernel_size=(7,7),padding=(3,3)),
            nn.Tanh()
        )

    def forward(self,z,c):
        """
        :param z: (batch_size,100)
        :param c: (batch_size,24)
        :return: (batch_size,3,64,64) tensor with value between [-1,+1]
        """
        condition=self.condition2img(c).view(-1,1,self.feature_map_size,self.feature_map_size)
        img=self.z2img(z).view(-1,512,self.feature_map_size,self.feature_map_size)
        out=torch.cat((condition,img),dim=1)  # become(N,513,8,8)
        out=self.convT1(out)  # become(N,256,16,16)
        out=self.convT2(out)  # become(N,128,32,32)
        out=self.convT3(out)  # become(N,128,64,64)
        out=self.conv(out)  # become(N,3,64,64)
        return out








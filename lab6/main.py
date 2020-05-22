import os
import torch
from torch.utils.data import DataLoader

from datahelper import CLEVRDataset
from model import Generator,Discriminator
from train import train


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim=100
c_dim=100
image_shape=(64,64,3)
epochs=1000
lr=0.0002
batch_size=64
LAMBDA=10   # gradient penalty weight in WGAN-GP
n_critic=5  # number of training steps for discriminator per iter

if __name__=='__main__':

    # load training data
    dataset_train=CLEVRDataset(img_path='iclevr',json_path=os.path.join('dataset','train.json'))
    loader_train=DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=4)

    # create generate & discriminator
    generator=Generator(z_dim,c_dim).to(device)
    discrimiator=Discriminator(image_shape,c_dim).to(device)
    # no need to init in WGAN-GP
    #generator.weight_init(0,0.02)
    #discrimiator.weight_init(0,0.02)

    # train
    train(loader_train,generator,discrimiator,z_dim,epochs,lr,LAMBDA,n_critic)


from datahelper import MyDataSet
from torch.utils.data import DataLoader
from model import VAE


dataset_train=MyDataSet('train.txt',is_train=True)
loader_train=DataLoader(dataset_train,batch_size=1,shuffle=True)
"""
for word1_tensor,tense1_tensor in loader_train:
    print(word1_tensor)
    print(tense1_tensor)

print example:
tensor([[[ 4],
         [11],
         [22],
         [ 7],
         [ 1]]])
tensor([[0]])
"""
vae=VAE(input_size=29,hidden_size=512,latent_size=32,conditional_size=8,max_length=dataset_train.max_length)
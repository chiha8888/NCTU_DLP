from datahelper import MyDataSet
from torch.utils.data import DataLoader
from model import VAE
from util import *

"""
dataset_train=MyDataSet('train.txt',is_train=True)
loader_train=DataLoader(dataset_train,batch_size=1,shuffle=True)

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
"""
klloss_tf_list=[0.1,0.2,0.3,0.10,0.9,0.8,0.7,0.3,0.1,0.1]
celoss_tf_list=[0.10,0.9,0.7,0.4,0.2,0.1,0.1,0.5,0.5,0.2]
BLEU_tf_list=[0.2,0.3,0.4,0.6,0.8,0.9,0.4,0.3,0.2,0.2]
gaussian_tf_list=[0,0,0,0.2,0.4,0.1,0,0,0,0.1]
teacher=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0]
kl=[0,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
fig=plot(10,celoss_tf_list,klloss_tf_list,BLEU_tf_list,gaussian_tf_list,teacher,kl)
fig.show()
plt.waitforbuttonpress(0)
"""


tf_list=[]
epochs=500

kl_weight_list=[]
kl_annealing_type='cycle'
time=2
for epoch in range(1,epochs+1):
    kl_weight_list.append(get_kl_weight(epoch,epochs,kl_annealing_type,time))
plt.plot(range(1,epochs+1),kl_weight_list,linestyle=':')


plt.show()

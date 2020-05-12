from __future__ import unicode_literals, print_function, division
import os
import torch
from torch import optim
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datahelper import MyDataSet
from model import VAE
from train import train,evaluate,generateWord
from util import get_teacher_forcing_ratio,get_kl_weight,get_gaussian_score,plot


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
#----------Hyper Parameters----------#
input_size = 28    # The number of vocabulary: input_size==vocab_size  {SOS,EOS,a,b,c,....,z}
hidden_size = 256  # LSTM hidden size
latent_size = 32
conditional_size = 8
LR = 0.05
epochs = 500
kl_annealing_type='cycle'  # 'monotonic' or 'cycle'
time = 2
#if('monotonic'): time is # of epoch for kl_weight from 0.0 to reach 1.0
#if('cycle'):     time is # of cycle


if __name__=='__main__':

    # dataloader
    dataset_train=MyDataSet(path='train.txt',is_train=True)
    loader_train=DataLoader(dataset_train,batch_size=1,shuffle=True)
    dataset_test=MyDataSet(path='test.txt',is_train=False)
    loader_test=DataLoader(dataset_test,batch_size=1,shuffle=False)

    print(f'MAX_LENGTH: {dataset_train.max_length}')
    # VAE model
    vae=VAE(input_size,hidden_size,latent_size,conditional_size,max_length=dataset_train.max_length).to(device)

    # train
    optimizer = optim.SGD(vae.parameters(), lr=LR)
    CEloss_list,KLloss_list,BLEUscore_list,teacher_forcing_ratio_list,kl_weight_list=[],[],[],[],[]
    best_BLEUscore=0
    best_model_wts=None
    for epoch in range(1,epochs+1):
        """
        train
        """
        # get teacher_forcing_ratio & kl_weight
        teacher_forcing_ratio=get_teacher_forcing_ratio(epoch,epochs)
        kl_weight=get_kl_weight(epoch,epochs,kl_annealing_type,time)
        CEloss,KLloss,_=train(vae, loader_train, optimizer, teacher_forcing_ratio, kl_weight, dataset_train.tensor2string)
        CEloss_list.append(CEloss)
        KLloss_list.append(KLloss)
        teacher_forcing_ratio_list.append(teacher_forcing_ratio)
        kl_weight_list.append(kl_weight)
        print(f'epoch{epoch:>2d}/{epochs}  tf_ratio:{teacher_forcing_ratio:.2f}  kl_weight:{kl_weight:.2f}')
        print(f'CE:{CEloss:.4f} + KL:{KLloss:.4f} = {CEloss+KLloss:.4f}')

        """
        evaluate
        """
        conversion,BLEUscore=evaluate(vae,loader_test,dataset_test.tensor2string)
        # generate words
        #generated_words=generateWord(vae,latent_size,dataset_test.tensor2string)
        #Gaussianscore=get_gaussian_score(generated_words)
        BLEUscore_list.append(BLEUscore)
        print(conversion)
        #print(generated_words)
        print(f'BLEU socre:{BLEUscore:.4f}') # Gaussian score:{Gaussianscore:.4f}')
        print()

        """
        update best model wts
        """
        if BLEUscore>best_BLEUscore:
            best_BLEUscore=BLEUscore
            best_model_wts=copy.deepcopy(vae.state_dict())
            # save model
            torch.save(best_model_wts,os.path.join('models',f'{kl_annealing_type}_time{time}_epochs{epochs}.pt'))
            fig=plot(epoch,CEloss_list,KLloss_list,BLEUscore_list,teacher_forcing_ratio_list,kl_weight_list)
            fig.savefig(os.path.join('results',f'{kl_annealing_type}_time{time}_epochs{epochs}.png'))
        
    torch.save(best_model_wts,os.path.join('models',f'{kl_annealing_type}_time{time}_epochs{epochs}.pt'))
    fig=plot(epochs,CEloss_list,KLloss_list,BLEUscore_list,teacher_forcing_ratio_list,kl_weight_list)
    fig.savefig(os.path.join('results',f'{kl_annealing_type}_time{time}_epochs{epochs}.png'))

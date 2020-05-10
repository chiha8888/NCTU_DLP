from __future__ import unicode_literals, print_function, division
import os
import torch
from torch import optim
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datahelper import MyDataSet
from model import VAE
from train import train,evaluate
from util import get_teacher_forcing_ratio,get_kl_weight,get_gaussian_score,generateWord,plot


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
#----------Hyper Parameters----------#
input_size = 29    # The number of vocabulary: input_size==vocab_size  {SOS,EOS,SPC,a,b,c,....,z}
hidden_size = 256  # LSTM hidden size
latent_size = 32
conditional_size = 8
LR = 0.05
epochs = 200
period=50
highest_tf=0.8
lowest_tf=0.0
kl_annealing_type='monotonic'  # 'monotonic' or 'cycle'


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
    CEloss_list,KLloss_list,BLEUscore_list,Gaussianscore_list,teacher_forcing_ratio_list,kl_weight_list=[],[],[],[],[],[]
    best_score=0  # score = BLEUscore + Gaussianscore
    best_model_wts=None
    for epoch in range(1,epochs+1):
        """
        train
        """
        # get teacher_forcing_ratio & kl_weight
        teacher_forcing_ratio=get_teacher_forcing_ratio(epoch,epochs,high=highest_tf,low=lowest_tf)
        kl_weight=get_kl_weight(epoch,kl_annealing_type,p=period)
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
        generated_words=generateWord(vae,latent_size,dataset_test.tensor2string)
        Gaussianscore=get_gaussian_score(generated_words)
        BLEUscore_list.append(BLEUscore)
        Gaussianscore_list.append(Gaussianscore)
        print(conversion)
        print(f'BLEU socre: {BLEUscore:.4f}')
        print(f'Gaussian score: {Gaussianscore:.4f}')
        print()

        """
        update best model wts
        """
        if BLEUscore+Gaussianscore>best_score:
            best_score=BLEUscore+Gaussianscore
            best_model_wts=copy.deepcopy(vae.state_dict())

    # save model
    torch.save(best_model_wts,os.path.join('models',f'{kl_annealing_type}_p{period}_score{best_score:.2f}.pt'))

    fig=plot(epochs,CEloss_list,KLloss_list,BLEUscore_list,Gaussianscore_list,teacher_forcing_ratio_list,kl_weight_list)
    fig.savefig(os.path.join('results',f'score{best_score:.2f}.png'))
    fig.show()
    plt.waitforbuttonpress(0)

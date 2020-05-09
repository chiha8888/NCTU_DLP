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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
#----------Hyper Parameters----------#
input_size = 29    # The number of vocabulary: input_size==vocab_size  {SOS,EOS,SPC,a,b,c,....,z}
hidden_size = 256  # LSTM hidden size
latent_size = 32
conditional_size = 8
LR = 0.05
epochs = 1000


def Gaussian_score(words):
    """
    :param words:
    words = [['consult', 'consults', 'consulting', 'consulted'],
             ['plead', 'pleads', 'pleading', 'pleaded'],
             ['explain', 'explains', 'explaining', 'explained'],
             ['amuse', 'amuses', 'amusing', 'amused'], ....]
    """
    words_list = []
    score = 0
    yourpath = os.path.join('dataset','train.txt')  #should be your directory of train.txt
    with open(yourpath,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)

def get_teacher_forcing_ratio(epoch,total_epoch):
    # range between 0.0~1.0
    return 1.-(1./total_epoch)*epoch

def get_kl_weight(epoch,kl_annealing_type,period):
    """
    :param epoch: i-th epoch
    :param kl_annealing_type: 'monotonic' or 'cyclical'
    :param period: #epoch for kl_weight to reach 1.0 from 0.0
    :return: 0.0~1.0 kl_weight
    """
    assert kl_annealing_type=='monotonic' or kl_annealing_type=='cyclical','kl_annealing_type not exist!'
    if kl_annealing_type=='monotonic':
        if epoch>=period:
            return 1.
        else:
            return (1./period)*epoch
    else:  # cyclical
        if (epoch//period)%2==1:
            return 1.
        else:
            return (1./period)*(epoch%period)


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
    CEloss_list=[]
    KLloss_list=[]
    BLEU_list=[]
    optimizer = optim.SGD(vae.parameters(), lr=LR)
    for epoch in range(1,epochs+1):
        """
        train
        """
        teacher_forcing_ratio=get_teacher_forcing_ratio(epoch,epochs)
        kl_weight=get_kl_weight(epoch,kl_annealing_type='monotonic',period=epochs)
        CEloss,KLloss,BLEUscore=train(vae, loader_train, optimizer, teacher_forcing_ratio, kl_weight, dataset_train.tensor2string)
        CEloss_list.append(CEloss)
        KLloss_list.append(KLloss)
        BLEU_list.append(BLEUscore)
        print(f'epoch{epoch:>2d}/{epochs}  teacher_forcing_ratio:{teacher_forcing_ratio:.2f}  kl_weight:{kl_weight:.2f}')
        print(f'CE:{CEloss:.4f} + KL:{KLloss:.4f} = {CEloss+KLloss:.4f}    {BLEUscore:.4f}')

        """
        evaluate
        """
        conversion,BLUEscore=evaluate(vae,loader_test,dataset_test.tensor2string)
        print(f'eval BLEU socre: {BLEUscore:.4f}')
        print(conversion)

        """
        Gaussian noise generation
        """
        gaussian_predict=[]
        max_gaussian_score=0
        for i in range(10):
            latent = torch.randn(1, 1, latent_size).to(device)
            predict=[]
            for tense in range(4):
                predict_output=dataset_test.tensor2string(vae.generate(latent,tense))
                predict.append(predict_output)
            gaussian_predict.append(predict)
        score=Gaussian_score(gaussian_predict)
        print(f'gaussian_score: {score:.4f}')
        max_gaussian_score=max(max_gaussian_score,score)

        print('==============================================================')

    print(f'max_gaussian_score={max_gaussian_score}')
    fig=plt.figure(figsize=(8,6))
    plt.plot(CEloss_list,label='CEloss')
    plt.plot(KLloss_list,label='KLloss')
    plt.legend()
    fig.show()
    fig.savefig('result.png')
    plt.waitforbuttonpress(0)

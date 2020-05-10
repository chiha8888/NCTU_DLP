import os
import torch
import matplotlib.pyplot as plt
device=device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_teacher_forcing_ratio(epoch,epochs,high,low):
    """map high~to from 1~epochs
    :param high: 0~1 float
    :param low: 0~1 float
    """
    return high+(low-high)/(epochs-1)*(epoch-1)  # y=mx+b

def get_kl_weight(epoch,kl_annealing_type,p):
    """
    :param epoch: i-th epoch
    :param kl_annealing_type: 'monotonic' or 'cycle'
    :param p: #epoch for kl_weight from 0.0 to reach 1.0
    :return: 0.0~1.0 kl_weight
    """
    assert kl_annealing_type=='monotonic' or kl_annealing_type=='cycle','kl_annealing_type not exist!'

    if kl_annealing_type == 'monotonic':
        return (1./(p-1))*(epoch-1) if epoch<p else 1.

    else:  # cycle
        return (1./(p-1))*(epoch%p) if (epoch//p)%2==0 else 1.

def get_gaussian_score(words):
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

def generateWord(vae,latent_size,tensor2string):
    vae.eval()
    re=[]
    for i in range(10):
        latent = torch.randn(1, 1, latent_size).to(device)
        tmp = []
        for tense in range(4):
            word = tensor2string(vae.generate(latent, tense))
            tmp.append(word)

        re.append(tmp)
    return re

def plot(epochs,CEloss_list,KLloss_list,BLEUscore_list,Gaussianscore_list,teacher_forcing_ratio_list,kl_weight_list):
    """
    CEloss_list,KLloss_list: from train
    BLEUscore_list,gaussian_score_list: from evaluate
    teacher_forcing_ratio_list,kl_weight_list: from user setting
    """
    x=range(1,epochs+1)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x,CEloss_list, label='CEloss')
    plt.plot(x,KLloss_list, label='KLloss')

    plt.plot(x,BLEUscore_list,'ro',label='BLEU score')
    plt.plot(x,Gaussianscore_list, 'bo', label='Gaussian score')
    plt.plot(x,[a+b for a,b in zip(BLEUscore_list,Gaussianscore_list)],linestyle='-.',label='BLEU+Gaussian score')

    plt.plot(x,teacher_forcing_ratio_list,linestyle=':',label='tf_ratio')
    plt.plot(x,kl_weight_list,linestyle=':',label='kl_weight')
    plt.legend()

    return fig



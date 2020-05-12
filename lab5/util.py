import os
import numpy as np
import torch
import matplotlib.pyplot as plt
device=device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_teacher_forcing_ratio(epoch,epochs):
    # from 1.0 to 0.0
    teacher_forcing_ratio = 1.-(1./(epochs-1))*(epoch-1)
    return teacher_forcing_ratio

def get_kl_weight(epoch,epochs,kl_annealing_type,time):
    """
    :param epoch: i-th epoch
    :param kl_annealing_type: 'monotonic' or 'cycle'
    :param time:
        if('monotonic'): # of epoch for kl_weight from 0.0 to reach 1.0
        if('cycle'):     # of cycle
    """
    assert kl_annealing_type=='monotonic' or kl_annealing_type=='cycle','kl_annealing_type not exist!'

    if kl_annealing_type == 'monotonic':
        return (1./(time-1))*(epoch-1) if epoch<time else 1.

    else: #cycle
        period = epochs//time
        epoch %= period
        KL_weight = sigmoid((epoch - period // 2) / (period // 10)) / 2
        return KL_weight

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

def plot(epochs,CEloss_list,KLloss_list,BLEUscore_list,teacher_forcing_ratio_list,kl_weight_list):
    """
    CEloss_list,KLloss_list: from train
    BLEUscore_list: from evaluate
    teacher_forcing_ratio_list,kl_weight_list: from user setting
    """
    x=range(1,epochs+1)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x,CEloss_list, label='CEloss')
    plt.plot(x,KLloss_list, label='KLloss')

    plt.plot(x,BLEUscore_list,label='BLEU score')

    plt.plot(x,teacher_forcing_ratio_list,linestyle=':',label='tf_ratio')
    plt.plot(x,kl_weight_list,linestyle=':',label='kl_weight')
    plt.legend()

    return fig



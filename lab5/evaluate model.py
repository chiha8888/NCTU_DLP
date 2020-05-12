from __future__ import unicode_literals, print_function, division
import os
import torch
from torch import optim
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datahelper import MyDataSet
from model import VAE
from train import train, evaluate, generateWord
from util import get_teacher_forcing_ratio, get_kl_weight, get_gaussian_score, plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
# ----------Hyper Parameters----------#
input_size = 28  # The number of vocabulary: input_size==vocab_size  {SOS,EOS,a,b,c,....,z}
hidden_size = 256  # LSTM hidden size
latent_size = 32
conditional_size = 8
max_length=15
file_path= 'best_cycle_time2_epochs500.pt'  #'score0.84.pt'

if __name__ == '__main__':
    # dataloader
    dataset_test = MyDataSet(path='test.txt', is_train=False)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # VAE model
    vae = VAE(input_size, hidden_size, latent_size, conditional_size, max_length).to(device)
    vae.load_state_dict(torch.load(os.path.join('models',file_path)))

    """
    evaluate
    """
    total_BLEUscore=0
    total_Gaussianscore=0
    test_time=20
    for i in range(test_time):
        conversion, BLEUscore = evaluate(vae, loader_test, dataset_test.tensor2string)
        # generate words
        generated_words=generateWord(vae,latent_size,dataset_test.tensor2string)
        Gaussianscore=get_gaussian_score(generated_words)
        print('test.txt prediction:')
        print(conversion)
        print('generate 100 words with 4 different tenses:')
        print(generated_words)
        print(f'BLEU socre:{BLEUscore:.2f}')
        print(f'Gaussian score:{Gaussianscore:.2f}')
        total_BLEUscore+=BLEUscore
        total_Gaussianscore+=Gaussianscore
    print()
    print(f'avg BLEUscore {total_BLEUscore/test_time:.2f}')
    print(f'avg Gaussianscore {total_Gaussianscore/test_time:.2f}')

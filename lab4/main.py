from __future__ import unicode_literals, print_function, division
import os
import torch
import copy
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from model import EncoderRNN,SimpleDecoderRNN,AttentionDecoderRNN
from train import trainIters,evaluateAll
from datahelper import DataTransformer
from plot import plot

"""========================================================================================
The main.py includes the following template functions:

1. Encoder, decoder
2. Training function
3. BLEU-4 score function

You have to modify them to complete the lab.
In addition, there are still other functions that you have to 
implement by yourself.

1. Your own dataloader (design in your own way, not necessary Pytorch Dataloader)
2. Output your results (BLEU-4 score, correction words)
3. Plot loss/score
4. Load/save weights 
========================================================================================"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
#----------Hyper Parameters----------#
hidden_size = 512  # LSTM hidden size
vocab_size = 29  # The number of vocabulary:vocab_size==input_size ,containing:SOS,EOS,UNK,a-z
teacher_forcing_ratio = 0.5
LR = 0.05
epochs = 50
decoder_type='simple'

#compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

if __name__=='__main__':
    """
    load training data
    """
    # training data
    datatransformer = DataTransformer()
    training_list,_ = datatransformer.build_training_set(path='train.json')
    training_tensor_list = []
    # convert list to tensor
    for training_pair in training_list:
        input_tensor = torch.tensor(training_pair[0], device=device).view(-1, 1)
        target_tensor = torch.tensor(training_pair[1], device=device).view(-1, 1)
        training_tensor_list.append((input_tensor, target_tensor))
    # testing data
    testing_list,testing_input=datatransformer.build_training_set(path='test.json')
    testing_tensor_list=[]
    # convert list to tensor
    for testing_pair in testing_list:
        input_tensor=torch.tensor(testing_pair[0],device=device).view(-1,1)
        target_tensor=torch.tensor(testing_pair[1],device=device).view(-1,1)
        testing_tensor_list.append((input_tensor,target_tensor))
    """
    model
    """
    encoder=EncoderRNN(vocab_size,hidden_size).to(device)
    if decoder_type=='simple':
        decoder=SimpleDecoderRNN(vocab_size,hidden_size).to(device)
    else:
        decoder=AttentionDecoderRNN(vocab_size,hidden_size,datatransformer.MAX_LENGTH).to(device)
    """
    train
    """
    loss_list=[]
    BLEU_list=[]
    best_score=0
    best_encoder_wts,best_decoder_wts=None,None
    for epoch in range(1,epochs+1):
        loss=trainIters(decoder_type,encoder,decoder,training_tensor_list,learning_rate=0.05,max_length=datatransformer.MAX_LENGTH,teacher_forcing_ratio=0.5,device=device)
        print(f'epoch{epoch:>2d} loss:{loss:.4f}')
        predicted_list=evaluateAll(decoder_type,encoder,decoder,testing_tensor_list,max_length=datatransformer.MAX_LENGTH,device=device)
        # test all testing data
        score=0
        for i,(input,target) in enumerate(testing_input):
            predict=datatransformer.indices2sequence(predicted_list[i])
            print(f'input:  {input}')
            print(f'target: {target}')
            print(f'pred:   {predict}')
            print('============================')
            score+=compute_bleu(predict,target)
        score/=len(testing_input)
        print(f'BLEU-4: {score:.2f}')

        loss_list.append(loss)
        BLEU_list.append(score)
        # update best model wts
        if score>best_score:
            best_score=score
            best_encoder_wts=copy.deepcopy(encoder.state_dict())
            best_decoder_wts=copy.deepcopy(decoder.state_dict())

    # save model
    torch.save(best_encoder_wts,os.path.join('models',f'encoder_teacher{teacher_forcing_ratio:.2f}_hidden{hidden_size}.pt'))
    torch.save(best_decoder_wts,os.path.join('models',f'{decoder_type}decoder_teacher{teacher_forcing_ratio:.2f}_hidden{hidden_size}.pt'))
    # plot
    figure=plot(loss_list,BLEU_list)
    figure.show()
    figure.savefig(os.path.join('result',f'{decoder_type}_teacher{teacher_forcing_ratio:.2f}_hidden{hidden_size}.png'))
    plt.waitforbuttonpress(0)





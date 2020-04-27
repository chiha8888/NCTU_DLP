from __future__ import unicode_literals, print_function, division
import os
import torch
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from model import EncoderRNN,SimpleDecoderRNN,AttentionDecoderRNN
from train import trainIters,evaluateAll
from datahelper import DataTransformer

#compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)


if __name__=='__main__':
    vocab_size=29
    hidden_size=512
    teacher_forcing_ratio=0.7
    decoder_type='simple'
    encoder_model_name=f'encoder_teacher{teacher_forcing_ratio:.2f}_hidden{hidden_size}.pt'
    decoder_model_name=f'{decoder_type}decoder_teacher{teacher_forcing_ratio:.2f}_hidden{hidden_size}.pt'
    """
    load testing data
    """
    datatransformer=DataTransformer()
    # testing data
    testing_list,testing_input=datatransformer.build_training_set(path='test.json')
    testing_tensor_list=[]
    # convert list to tensor
    for testing_pair in testing_list:
        input_tensor=torch.tensor(testing_pair[0],device=device).view(-1,1)
        target_tensor=torch.tensor(testing_pair[1],device=device).view(-1,1)
        testing_tensor_list.append((input_tensor,target_tensor))
    """
    load model
    """
    encoder=EncoderRNN(vocab_size,hidden_size).to(device)
    encoder.load_state_dict(torch.load(os.path.join('models',encoder_model_name)))
    decoder=SimpleDecoderRNN(vocab_size,hidden_size).to(device)
    decoder.load_state_dict(torch.load(os.path.join('models',decoder_model_name)))
    """
    test
    """
    predicted_list = evaluateAll(decoder_type, encoder, decoder, testing_tensor_list,max_length=20, device=device)
    # test all testing data
    score = 0
    for i, (input, target) in enumerate(testing_input):
        predict = datatransformer.indices2sequence(predicted_list[i])
        print(f'input:  {input}')
        print(f'target: {target}')
        print(f'pred:   {predict}')
        print('============================')
        score += compute_bleu(predict, target)
    score /= len(testing_input)
    print(f'BLEU-4: {score:.2f}')

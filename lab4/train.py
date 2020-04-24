from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
from torch import optim

SOS_token=0
EOS_token=1

def train(decoder_type,input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,max_length,teacher_forcing_ratio,device):
    """
    train by one (input,target) pair
    :param decoder_type: 'simple' or 'attention'
    :param input_tensor: (time1,1) tensor for encoder
    :param target_tensor: (time2,1) tensor for decoder
    :param max_length: word maximum length in training data
    :return: loss value
    """
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    loss = 0
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    """
    encoder forwarding
    """
    encoder_hidden_state=encoder.init_h0()
    encoder_cell_state=encoder.init_c0()
    for ei in range(input_length):
        encoder_output,encoder_hidden_state,encoder_cell_state=encoder(input_tensor[ei],encoder_hidden_state,encoder_cell_state)
        # encoder_output: (time,batch,num_directions*hidden_size)
        encoder_outputs[ei]=encoder_outputs[0,0]

    """
    decoder forwarding
    """
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden_state=encoder_hidden_state
    decoder_cell_state=encoder_cell_state

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if decoder_type=='simple':
                decoder_output,decoder_hidden_state,decoder_cell_state=decoder(decoder_input,decoder_hidden_state,decoder_cell_state)
            else:  # decoder_type=='attention'
                decoder_output,decoder_hidden_state,decoder_cell_state,_=decoder(decoder_input,decoder_hidden_state,decoder_cell_state,encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])

            decoder_input = target_tensor[di]  # Teacher forcing
            # don't care decoder_output is EOS or not

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if decoder_type == 'simple':
                decoder_output, decoder_hidden_state, decoder_cell_state = decoder(decoder_input,decoder_hidden_state,decoder_cell_state)
            else:  # decoder_type=='attention'
                decoder_output, decoder_hidden_state, decoder_cell_state, _ = decoder(decoder_input,decoder_hidden_state,decoder_cell_state,encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()/target_length

def trainIters(decoder_type,encoder,decoder,training_pairs,learning_rate,max_length,teacher_forcing_ratio,device):
    """
    :param decoder_type: 'simple' or 'attention'
    :param training_pairs: [(input,target),(input,target)....(input,target)] (input=(input_len,1)tensor, target=(target_len,1)tensor)
    """
    assert decoder_type=='simple' or decoder_type=='attention','no such decoder_type'
    loss_total=0
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    random.shuffle(training_pairs)  # shuffle training_pairs
    for input_tensor,target_tensor in training_pairs:
        loss = train(decoder_type,input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,max_length,teacher_forcing_ratio,device)
        loss_total+=loss

    return loss_total/len(training_pairs)

def evaluate(decoder_type,input_tensor,encoder,decoder,max_length,device):
    """
    :param decoder_type: 'simple' or 'attention'
    :param input_tensor: (time,1) tensor for encoder
    :param max_length: word maximum length in training data
    :return: predicted indices list
    """
    predicted=[]
    input_length=input_tensor.size(0)
    encoder_outputs=torch.zeros(max_length,encoder.hidden_size,device=device)
    """
    encoder forwarding
    """
    encoder_hidden_state=encoder.init_h0()
    encoder_cell_state=encoder.init_c0()
    for ei in range(input_length):
        encoder_output,encoder_hidden_state,encoder_cell_state=encoder(input_tensor[ei],encoder_hidden_state,encoder_cell_state)
        encoder_outputs[ei]=encoder_output[0,0]
    """
    decoder forwarding
    """
    decoder_input=torch.tensor([[SOS_token]],device=device)
    decoder_hidden_state=encoder_hidden_state
    decoder_cell_state=encoder_cell_state
    for di in range(max_length):
        if decoder_type=='simple':
            decoder_output,decoder_hidden_state,decoder_cell_state=decoder(decoder_input,decoder_hidden_state,decoder_cell_state)
        else:  # decoder_type=='attention'
            decoder_output,decoder_hidden_state,decoder_cell_state,_=decoder(decoder_input,decoder_hidden_state,decoder_cell_state,encoder_outputs)
        topv,topi=decoder_output.data.topk(1)
        decoder_input=topi.squeeze().detach()
        if decoder_input.item()==EOS_token:
            break
        else:
            predicted.append(decoder_input.item())
    return predicted

def evaluateAll(decoder_type,encoder,decoder,testing_pairs,max_length,device):
    """
    :param decoder_type: 'simple' or 'attention'
    :param testing_pairs: [(input,target),(input,target)....(input,target)] (input=(input_len,1)tensor, target=(target_len,1)tensor)
    :param max_length: word maximum length in training data
    :return: [predicted1,predicted2,......] a list of integer list
    """
    assert decoder_type=='simple' or decoder_type=='attention','no such decoder_type'
    predicted_list=[]
    for input_tensor,target_tensor in testing_pairs:
        predicted_list.append(evaluate(decoder_type,input_tensor,encoder,decoder,max_length,device))
    return predicted_list
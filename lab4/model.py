from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Encoder
class EncoderRNN(nn.Module):
    def __init__(self,input_size,hidden_size):
        """
        output of rnn is not used in simple decoder,but used in attention decoder
        :param input_size: 29 (containing:SOS,EOS,UNK,a-z)
        :param hidden_size: 256
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden_state, cell_state):
        """
        batch_size here is 1
        :param input: tensor
        :param hidden_state: (num_layers*num_directions=1,batch_size=1,vec_dim=256)
        :param cell_state: (num_layers*num_directions=1,batch_size=1,vec_dim=256)
        """
        embedded = self.embedding(input).view(1, 1, -1)  # view(1,1,-1) due to input of rnn must be (seq_len,batch,vec_dim)
        output,(hidden_state,cell_state) = self.rnn(embedded, (hidden_state,cell_state) )
        return output,hidden_state,cell_state

    def init_h0(self):
        """
        :return: (num_layers * num_directions, batch, hidden_size)
        """
        return torch.zeros(1, 1, self.hidden_size, device=device)
    def init_c0(self):
        """
        :return: (num_layers * num_directions, batch, hidden_size)
        """
        return torch.zeros(1, 1, self.hidden_size, device=device)

#Simple Decoder
class SimpleDecoderRNN(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(SimpleDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, input_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden_state, cell_state):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, (hidden_state,cell_state) = self.rnn(output, (hidden_state,cell_state) )
        output = self.softmax(self.out(output[0]))
        return output,hidden_state,cell_state

    def init_h0(self):
        """
        :return: (num_layers * num_directions, batch, hidden_size)
        """
        return torch.zeros(1, 1, self.hidden_size, device=device)
    def init_c0(self):
        """
        :return: (num_layers * num_directions, batch, hidden_size)
        """
        return torch.zeros(1, 1, self.hidden_size, device=device)


#Attention Decoder
class AttentionDecoderRNN(nn.Module):
    def __init__(self,input_size,hidden_size,max_length,dropout_p=0.1):
        """
        :param input_size: 29
        :param hidden_size: 256
        """
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, input, hidden_state, cell_state, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden_state[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, (hidden_state,cell_state) = self.rnn(output, (hidden_state,cell_state) )

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output,hidden_state,cell_state,attn_weights

    def init_h0(self):
        """
        :return: (num_layers * num_directions, batch, hidden_size)
        """
        return torch.zeros(1, 1, self.hidden_size, device=device)
    def init_c0(self):
        """
        :return: (num_layers * num_directions, batch, hidden_size)
        """
        return torch.zeros(1,1,self.hidden_size,device=device)

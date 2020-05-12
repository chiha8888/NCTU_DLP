from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token=0
EOS_token=1

class VAE(nn.Module):  # conditional VAE
    # Encoder
    class EncoderRNN(nn.Module):
        def __init__(self, input_size, hidden_size):
            """
            :param input_size: 28 (containing:SOS,EOS,a-z)
            :param hidden_size: 256 or 512
            """
            super(VAE.EncoderRNN,self).__init__()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.rnn = nn.LSTM(hidden_size, hidden_size)

        def forward(self, input, hidden_state, cell_state):
            """forwarding an alphabet (batch_size here is 1)
            :param input: tensor
            :param hidden_state: (num_layers*num_directions=1,batch_size=1,vec_dim=256)
            :param cell_state: (num_layers*num_directions=1,batch_size=1,vec_dim=256)
            """
            embedded = self.embedding(input).view(1,1,-1)  # view(1,1,-1) due to input of rnn must be (seq_len,batch,vec_dim)
            output, (hidden_state, cell_state) = self.rnn(embedded, (hidden_state, cell_state))
            return output, hidden_state, cell_state

        def init_h0(self,size):
            """
            :return: (num_layers * num_directions, batch, hidden_size)
            """
            return torch.zeros(1, 1, size, device=device)

        def init_c0(self):
            """
            :return: (num_layers * num_directions, batch, hidden_size)
            """
            return torch.zeros(1, 1, self.hidden_size, device=device)

    # Decoder
    class DecoderRNN(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(VAE.DecoderRNN, self).__init__()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.rnn = nn.LSTM(hidden_size, hidden_size)
            self.out = nn.Linear(hidden_size, input_size)
            self.softmax = nn.LogSoftmax(dim=1)

        def forward(self, input, hidden_state, cell_state):
            """forwarding an alphabet
            """
            output = self.embedding(input).view(1, 1, -1)
            output = F.relu(output)
            output, (hidden_state, cell_state) = self.rnn(output, (hidden_state, cell_state))
            output = self.softmax(self.out(output[0]))
            return output, hidden_state, cell_state

        def init_h0(self):
            pass

        def init_c0(self):
            """
            :return: (num_layers * num_directions, batch, hidden_size)
            """
            return torch.zeros(1, 1, self.hidden_size, device=device)

    def __init__(self, input_size, hidden_size, latent_size, conditional_size,max_length):
        """
        :param input_size: 28
        :param hidden_size: 256 or 512
        :param latent_size: 32
        :param conditional_size: 8
        """
        super(VAE,self).__init__()
        self.encoder = self.EncoderRNN(input_size, hidden_size)
        self.decoder = self.DecoderRNN(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.conditional_size = conditional_size
        self.max_length=max_length
        self.tense_embedding = nn.Embedding(4, conditional_size)  # 4 tense(simple present(sp), third person(tp), present progressive(pg), simple past(p))
        self.hidden2mean = nn.Linear(hidden_size, latent_size)
        self.hidden2logvar = nn.Linear(hidden_size, latent_size)
        self.latentcondition2hidden=nn.Linear(latent_size+conditional_size,hidden_size)

    def forward(self,input_tensor,encoder_hidden_state,encoder_cell_state,c,use_teacher_forcing):
        """forwarding a word in VAE (This is not training function)
        :param input_tensor: (in_tensor_length,1) tensor  (input_tensor==target_tensor)
        :param encoder_hidden_state: (num_layers * num_directions, batch, hidden_size) tensor
        :param encoder_cell_state: (num_layers * num_directions, batch, hidden_size)
        :param c: (1,1,conditional_size) tensor
        :param use_teacher_forcing: 0.0~1.0
        :returns:
            predict_output: (predict_length,1) tensor   (very likely contain EOS)
            predict_distribution: (in_tensor_length,input_size) tensor
            mean
            logvariance
        """
        """
        encoder forwarding
        """
        input_length=input_tensor.size(0)
        for ei in range(input_length):
            _ ,encoder_hidden_state,encoder_cell_state=self.encoder(input_tensor[ei],encoder_hidden_state,encoder_cell_state)
        """
        middle part forwarding
        """
        mean=self.hidden2mean(encoder_hidden_state)
        logvar=self.hidden2logvar(encoder_hidden_state)
        # sampling a point
        latent=self.reparameterize(mean,logvar)
        decoder_hidden_state = self.latentcondition2hidden(torch.cat((latent, c), dim=-1))
        decoder_cell_state = self.decoder.init_c0()
        decoder_input = torch.tensor([[SOS_token]], device=device)

        """
        decoder forwarding
        """
        predict_distribution=torch.zeros(input_length,self.input_size,device=device)
        predict_output = None
        for di in range(input_length):
            output,decoder_hidden_state,decoder_cell_state=self.decoder(decoder_input,decoder_hidden_state,decoder_cell_state)
            predict_distribution[di]=output[0]
            predict_class=output.topk(1)[1]
            predict_output=torch.cat((predict_output,predict_class)) if predict_output is not None else predict_class

            if use_teacher_forcing:  # use teacher forcing
                decoder_input=input_tensor[di]
            else:
                if predict_class.item() == EOS_token:
                    break
                decoder_input = predict_class

        return predict_output,predict_distribution,mean,logvar

    def inference(self,input_tensor,encoder_hidden_state,encoder_cell_state,c):
        """when "evaluation" forwarding a word in VAE
        :param input_tensor: (time1,1) tensor  (input_tensor==target_tensor)
        :param encoder_hidden_state: (num_layers * num_directions, batch, hidden_size) tensor
        :param encoder_cell_state: (num_layers * num_directions, batch, hidden_size)
        :param c: (1,1,conditional_size) tensor
        :return predict_output: (predict_output_length,1) tensor   (very likely contain EOS)
        """
        """
        encoder forwarding
        """
        input_length=input_tensor.size(0)
        for ei in range(input_length):
            _ ,encoder_hidden_state,encoder_cell_state=self.encoder(input_tensor[ei],encoder_hidden_state,encoder_cell_state)
        """
        middle part forwarding
        """
        mean=self.hidden2mean(encoder_hidden_state)
        logvar=self.hidden2logvar(encoder_hidden_state)
        # sampling a point
        latent=self.reparameterize(mean,logvar)
        decoder_hidden_state = self.latentcondition2hidden(torch.cat((latent, c), dim=-1))
        decoder_cell_state = self.decoder.init_c0()
        decoder_input = torch.tensor([[SOS_token]], device=device)

        """
        decoder forwarding
        """
        predict_output = None
        for di in range(self.max_length):
            output, decoder_hidden_state, decoder_cell_state = self.decoder(decoder_input,decoder_hidden_state,decoder_cell_state)
            predict_class = output.topk(1)[1]
            predict_output = torch.cat((predict_output, predict_class)) if predict_output is not None else predict_class

            if predict_class.item() == EOS_token:
                break
            decoder_input = predict_class

        return predict_output

    def generate(self,latent,tense):
        """
        :param latent: (1,1,latent_size) tensor
        :param tense: 0~3 int
        :return predict_output: (predict_output_length,1) tensor   (very likely contain EOS)
        """
        tense_tensor=torch.tensor([tense]).to(device)
        c=self.tense_embedding(tense_tensor).view(1, 1, -1)
        decoder_hidden_state = self.latentcondition2hidden(torch.cat((latent, c), dim=-1))
        decoder_cell_state=self.decoder.init_c0()
        decoder_input = torch.tensor([[SOS_token]], device=device)

        """
        decoder forwarding
        """
        predict_output = None
        for di in range(self.max_length):
            output, decoder_hidden_state, decoder_cell_state = self.decoder(decoder_input, decoder_hidden_state,
                                                                            decoder_cell_state)
            predict_class = output.topk(1)[1]
            predict_output = torch.cat((predict_output, predict_class)) if predict_output is not None else predict_class

            if predict_class.item() == EOS_token:
                break
            decoder_input = predict_class

        return predict_output

    def reparameterize(self,mean,logvar):
        """reparameterization trick
        """
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        latent=mean+eps*std
        return latent




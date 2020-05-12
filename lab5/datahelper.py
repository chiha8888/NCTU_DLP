import os
import torch
import torch.utils.data as data

class DataTransformer:
    def __init__(self):
        self.char2idx=self.build_char2idx()  # {'SOS':0,'EOS':1,'a':2,'b':3 ... 'z':27}
        self.idx2char=self.build_idx2char()  # {0:'SOS',1:'EOS',2:'a',3:'b' ... 27:'z'}
        self.tense2idx={'sp':0,'tp':1,'pg':2,'p':3}
        self.idx2tense={0:'sp',1:'tp',2:'pg',3:'p'}
        self.max_length=0  # max length of the training data word(contain 'EOS')

    def build_char2idx(self):
        dictionary={'SOS':0,'EOS':1}
        dictionary.update([(chr(i+97),i+2) for i in range(0,26)])
        return dictionary

    def build_idx2char(self):
        dictionary={0:'SOS',1:'EOS'}
        dictionary.update([(i+2,chr(i+97)) for i in range(0,26)])
        return dictionary

    def string2tensor(self,string,add_eos=True):
        """
        :param add_eox: True or False
        :return: (time1,1) tensor
        """
        indices=[self.char2idx[char] for char in string]
        if add_eos:
            indices.append(self.char2idx['EOS'])
        return torch.tensor(indices,dtype=torch.long).view(-1,1)

    def tense2tensor(self,tense):
        """
        :param tense: 0~3
        :return: (1) tensor
        """
        return torch.tensor([tense],dtype=torch.long)

    def tensor2string(self,tensor):
        """
        :param tensor: (time1,1) tensor
        :return: string (not contain 'EOS')
        """
        re=""
        string_length=tensor.size(0)
        for i in range(string_length):
            char=self.idx2char[tensor[i].item()]
            if char=='EOS':
                break
            re+=char
        return re

    def get_dataset(self,path,is_train):
        """
        :returns:
        if(train):  words=[w1,w2,w3,.....,wn], tenses:[0,1,2,3,0,1,2,3...]
        if(test):  words=[[w1,w2],[w3,w4]....[wn-1,wn]], tense:[[sp,p],[sp,pg]...,[pg,tp]]
        """
        words=[]
        tenses=[]
        with open(path,'r') as file:
            if is_train:
                for line in file:
                    words.extend(line.split('\n')[0].split(' '))
                    tenses.extend(range(0,4))
            else:
                for line in file:
                    words.append(line.split('\n')[0].split(' '))
                test_tenses=[['sp','p'],['sp','pg'],['sp','tp'],['sp','tp'],['p','tp'],['sp','pg'],['p','sp'],['pg','sp'],['pg','p'],['pg','tp']]
                for test_tense in test_tenses:
                    tenses.append([self.tense2idx[tense] for tense in test_tense])
        return words,tenses

class MyDataSet(data.Dataset):
    def __init__(self,path,is_train):
        self.is_train = is_train
        self.dataTransformer=DataTransformer()
        self.words,self.tenses=self.dataTransformer.get_dataset(os.path.join('dataset',path),is_train)
        self.max_length=self.get_max_length(self.words)
        self.string2tensor=self.dataTransformer.string2tensor  # output=(time1,1) tensor
        self.tense2tensor=self.dataTransformer.tense2tensor    # output=(1) tensor
        self.tensor2string=self.dataTransformer.tensor2string  # input=(time1,1) tensor
        assert len(self.words)==len(self.tenses),'word list is not compatible with tense list'

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        """
        :returns:
        if(train): word: (time1,1) tensor, tense: (1) tensor
        if(test): word1: (time1,1) tensor, tense1: (1) tensor, word2: (time2,1) tensor, tense2: (1) tensor
        """
        if self.is_train:
            return self.string2tensor(self.words[idx],add_eos=True),self.tense2tensor(self.tenses[idx])
        else:
            return self.string2tensor(self.words[idx][0],add_eos=True),self.tense2tensor(self.tenses[idx][0]),\
                   self.string2tensor(self.words[idx][1],add_eos=True),self.tense2tensor(self.tenses[idx][1])

    def get_max_length(self,words):
        max_length=0
        for word in words:
            max_length=max(max_length,len(word))
        return max_length
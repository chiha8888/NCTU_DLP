import json


class DataTransformer:
    def __init__(self):
        self.char2idx=self.build_char2idx()
        self.idx2char=self.build_idx2char()
        self.MAX_LENGTH=0  # max length of the training data word(contain 'EOS')

    def build_char2idx(self):
        """
        {'SOS':0,'EOS':1,'UNK':2,'a':3,'b':4 ... 'z':28}
        """
        dictionary={'SOS':0,'EOS':1,'UNK':2}
        dictionary.update([(chr(i+97),i+3) for i in range(0,26)])
        return dictionary

    def build_idx2char(self):
        """
        {0:'SOS',1:'EOS',2:'UNK',3:'a',4:'b' ... 28:'z'}
        """
        dictionary={0:'SOS',1:'EOS',2:'UNK'}
        dictionary.update([(i+3,chr(i+97)) for i in range(0,26)])
        return dictionary

    def sequence2indices(self,sequence,add_eos=True):
        """
        :param sequence(string): a char sequence
        :param add_eox(boolean): whether add 'EOS' at the end of the sequence
        :return: int sequence
        """
        indices=[]
        for c in sequence:
            indices.append(self.char2idx[c])
        if add_eos:
            indices.append(self.char2idx['EOS'])
        self.MAX_LENGTH = max(self.MAX_LENGTH, len(indices))
        return indices

    def indices2sequence(self,indices):
        """
        :param indices: int sequence (without EOS_token)
        :return: string
        """
        re=""
        for i in indices:
            re+=self.idx2char[i]
        return re

    def build_training_set(self,path):
        """
        :return:
            int_list: [[input,target],[input,target]....]  (input & target are all int sequence)
            str_list: [[input,target],[input,target]....]  (input & target are all string)
        """
        int_list=[]
        str_list=[]
        with open(path,'r') as file:
            dict_list=json.load(file)
            for dict in dict_list:
                target=self.sequence2indices(dict['target'])
                for input in dict['input']:
                    int_list.append([self.sequence2indices(input,add_eos=True),target])
                    str_list.append([input,dict['target']])
        return int_list,str_list
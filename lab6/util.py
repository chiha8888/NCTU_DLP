import os
import json
import torch

def get_test_conditions(path):
    """
    :return: (#test conditions,#classes) tensors
    """
    with open(os.path.join('dataset', 'objects.json'), 'r') as file:
        classes = json.load(file)
    with open(path,'r') as file:
        test_conditions_list=json.load(file)

    labels=torch.zeros(len(test_conditions_list),len(classes))
    for i in range(len(test_conditions_list)):
        for condition in test_conditions_list[i]:
            labels[i,int(classes[condition])]=1.

    return labels
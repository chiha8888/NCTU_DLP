from datahelper import DataTransformer

datatransformer=DataTransformer()
training_list=datatransformer.build_training_set(path='train.json')
print(datatransformer.MAX_LENGTH)




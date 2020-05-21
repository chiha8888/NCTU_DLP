This is the specification file for conditional GAN training and testing dataset - i-CLEVR

1. objects.json
This file is a dictionary file that contains the number of objects and the idexes.
There are totally 24 objects in i-CLEVR datasets with 3 shapes and 8 colors.

2. train.json
The file is for training. The number off training data is 18009.
train.json is a dictionary where keys are filenames and values are objects/
For example:
{"CLEVR_train_001032_0.png": ["yellow sphere"], 
"CLEVR_train_001032_1.png": ["yellow sphere", "gray cylinder"], 
"CLEVR_train_001032_2.png": ["yellow sphere", "gray cylinder", "purple cube"], ... }
One image can include objects from 1 to 3

3. test.json
The file is for testing. The number of testing data is 32.
test.json is a list where each element includes multiple objects
For example:
[['gray cube'],
['red cube'],
['blue cube'],
['blue cube', 'green cube'], ...]
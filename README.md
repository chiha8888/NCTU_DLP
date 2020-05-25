# NCTU_DLP
2020 NCTU資工所 Deep Learning and Practice class  
teacher:陳永昇、吳毅成、彭文孝  
## lab1
手刻bp  
參考：  
(1)https://www.youtube.com/watch?v=-yhm3WdGFok  
(2)https://www.brilliantcode.net/1381/backpropagation-2-forward-pass-backward-pass/?fbclid=IwAR063nI7xQA2gaUzgltwxr8a1cXc_sUmgGIDJuUE1GYrInjtWJbomEa4PXs  
## lab2
加入regularization term，acc: 82% -> 88%  
## lab3
經實驗發現:  
1. 無正規化、  
正規化 包括：transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):from Imagenet、
transforms.Normalize([0.5769, 0.3852, 0.2649],[0.1061, 0.0809, 0.0555]):from 眼球CenterCrop、transforms.Normalize([0.3749, 0.2602, 0.1857],[0.2526, 0.1780, 0.1291]):from 眼球training dataset  
acc都為82%  

2. 專注(transforms.CenterCrop(300))acc為80%  

3. weighted_loss([1.0, 10.565217391304348, 4.906175771971497, 29.591690544412607, 35.55077452667814])acc為80%，
但confusion matrix最為正確，最適合實際醫學應用  
## lab4
attention decoder BLEU score: 0.6(code應該有錯)  
simple decoder BLEU score: 0.98  
## lab5
test.txt prediction:  
[['abandon', 'abandoned', 'abandoned'], ['abet', 'abetting', 'abetting'], ['begin', 'begins', 'begins'], ['expend', 'expends', 'expects'], ['sent', 'sends', 'senses'], ['split', 'splitting', 'splitting'], ['flared', 'flare', 'flare'], ['functioning', 'function', 'function'], ['functioning', 'functioned', 'functioned'], ['healing', 'heals', 'heals']]  

generate words with 4 different tenses:  
[['applie', 'applies', 'applies', 'applied'], ['fester', 'festures', 'festering', 'festered'], ['realize', 'realizes', 'realing', 'realized'], ['allare', 'alleges', 'alleging', 'alleged'], ['pray', 'prattles', 'praying', 'prattled'], ['scrutinize', 'scrutinizes', 'scrutinizing', 'scrutinized'], ['pelmit', 'pellies', 'pellioning', 'expiated'], ['feil', 'feils', 'feiling', 'feilched'], ['repail', 'repairs', 'repairing', 'repaired'], ['jab', 'jabsts', 'jabbing', 'jabbed']]  

avg BLEUscore: 0.79  
avg Gaussianscore: 0.36  
## lab6
cDCGAN score: 0.70  
WGAN-GP score: 0.40  
傳統cDCGAN score: 0.36

結論：  
1. train G 4times 跟 train G 5times 結果差不多  
2. Generator要用RELU + Discriminator要用LeakyRelu 的效果好一點  
3. c_dim 200 比 100 好  

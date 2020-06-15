# NCTU_DLP
2020 NCTU資工所 Deep Learning and Practice class  
teacher:陳永昇、吳毅成、彭文孝  
## lab1
實做backpropagatioin  

參考：  
(1)https://www.youtube.com/watch?v=-yhm3WdGFok  
(2)https://www.brilliantcode.net/1381/backpropagation-2-forward-pass-backward-pass/?fbclid=IwAR063nI7xQA2gaUzgltwxr8a1cXc_sUmgGIDJuUE1GYrInjtWJbomEa4PXs  

## lab2
EEGNet & DeepConvNet for 2-classification problem  

加入regularization term後，acc: 82% -> 88%  

## lab3
image classification in 黃斑部病變  

經實驗發現:  
1. 無正規化、  
正規化 包括：transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):from Imagenet、
transforms.Normalize([0.5769, 0.3852, 0.2649],[0.1061, 0.0809, 0.0555]):from 眼球CenterCrop、transforms.Normalize([0.3749, 0.2602, 0.1857],[0.2526, 0.1780, 0.1291]):from 眼球training dataset  
acc都為82%  

2. 專注(transforms.CenterCrop(300))acc為80%  

3. weighted_loss([1.0, 10.565217391304348, 4.906175771971497, 29.591690544412607, 35.55077452667814])acc為80%，
但confusion matrix最為正確，最適合實際醫學應用  

## lab4
Seq2Seq Network solving "錯誤拼字English word" -> "正確拼字English word"  

attention decoder BLEU score: 0.6(code應該有錯)  
simple decoder BLEU score: 0.98  

## lab5
Conditional Seq2Seq VAE 轉 English word 的時態(sp,tp,pg,p)  

test.txt prediction:  
[['abandon', 'abandoned', 'abandoned'], ['abet', 'abetting', 'abetting'], ['begin', 'begins', 'begins'], ['expend', 'expends', 'expects'], ['sent', 'sends', 'senses'], ['split', 'splitting', 'splitting'], ['flared', 'flare', 'flare'], ['functioning', 'function', 'function'], ['functioning', 'functioned', 'functioned'], ['healing', 'heals', 'heals']]  

generate words with 4 different tenses:  
[['applie', 'applies', 'applies', 'applied'], ['fester', 'festures', 'festering', 'festered'], ['realize', 'realizes', 'realing', 'realized'], ['allare', 'alleges', 'alleging', 'alleged'], ['pray', 'prattles', 'praying', 'prattled'], ['scrutinize', 'scrutinizes', 'scrutinizing', 'scrutinized'], ['pelmit', 'pellies', 'pellioning', 'expiated'], ['feil', 'feils', 'feiling', 'feilched'], ['repail', 'repairs', 'repairing', 'repaired'], ['jab', 'jabsts', 'jabbing', 'jabbed']]  

avg BLEUscore: 0.79  
avg Gaussianscore: 0.36  

## lab6
GAN-based model生成具有有色幾何物體的圖片(64x64x3)  

cDCGAN score: 0.71  
WGAN-GP score: 0.40  
傳統cDCGAN score: 0.36
正解: ACGAN

結論：  
1. train G 4times 跟 train G 5times 結果差不多  
2. Generator要用RELU + Discriminator要用LeakyRelu  
3. 加入BN提高score  
4. c_dim 200 比 100 好  
5. 生成fake images時所用的condition vector用training data已有的condition就好了,自己隨機random的condition vector(24-dim中有1~3個1)反而會train壞掉  

## lab7
RL N-tuple Network解2048  

共有4(pattern)x8(isomorphism)=32個feature  
weights.bin in code dir:https://drive.google.com/file/d/116l-xMGBTbqJVKA1sYHQiHPbrQzeeemL/view?usp=sharing  

2048的win-rate: 94.1%  

## lab8
RL to learn LunarLander-v2(action離散) 與 LunarLanderContinuous-v2(action連續) Game  

實做DQN,DDQN,DDPG(Deep Deterministic Policy Gradient)  

DQN avg reward: 287.0  
DDQN avg reward: 274.4  
DDPG avg reward: 282.9  

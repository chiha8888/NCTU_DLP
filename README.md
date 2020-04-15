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
baseline:82%  
input size(512x512)+randomflip+normalize+feature extraction 5 epochs+finetuning 15 epochs  
專注：?  
input size(300x300)+randomflip+normalize+feature extraction 5 epochs+finetuning 15 epochs  
專注_無正規：？  
input size(300x300)+randomflip+feature extraction 5 epochs+finetuning 15 epochs  
weightedCrossEntropyLoss_無正規：壞掉...  
![weightedCrossEntropyLoss_無正規](https://i.imgur.com/gmjzRGD.png)

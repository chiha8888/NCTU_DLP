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
baseline:82.0%  
input size(512x512)+randomflip+normalize+feature extraction 5 epochs+finetuning 10 epochs  
![baseline](https://i.imgur.com/hSkWsaH.png)  
- - -  
無正規：81.9%  
input size(512x512)+randomflip+feature extraction 5 epochs+finetuning 15 epochs  
![無正規](https://i.imgur.com/FoDjK9l.png)  
- - -  
專注：?  
input size(300x300)+randomflip+normalize+feature extraction 5 epochs+finetuning 15 epochs  

- - -  
專注_無正規：80%  
input size(300x300)+randomflip+feature extraction 5 epochs+finetuning 15 epochs  
![專注_無正規](https://i.imgur.com/8k5uysM.png)  
- - -  
weightedCrossEntropyLoss_無正規：79%（不穩定）  
weight:[1.0, 10.565217391304348, 4.906175771971497, 29.591690544412607, 35.55077452667814]  
![weightedCrossEntropyLoss_無正規](https://i.imgur.com/gmjzRGD.png)  


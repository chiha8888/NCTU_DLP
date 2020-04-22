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


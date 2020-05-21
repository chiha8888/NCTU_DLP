import torch
import torch.nn as nn
import numpy as np
import copy
from util import get_test_conditions
from evaluator import EvaluationModel
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(dataloader,g_model,d_model,z_dim,c_dim,epochs,lr,test_path):
    Criterion=nn.CrossEntropyLoss()
    optimizer_g=torch.optim.Adam(g_model.parameters(),lr)
    optimizer_d=torch.optim.Adam(d_model.parameters(),lr)
    evaluation_model=EvaluationModel()

    test_conditions=get_test_conditions(test_path).to(device)
    best_score=0

    for epoch in range(1,1+epochs):
        total_loss_g=0
        total_loss_d=0
        for i,(images,conditions) in enumerate(dataloader):
            batch_size=len(images)
            images = images.to(device)
            conditions = conditions.to(device)
            real = torch.ones(batch_size, dtype=torch.long).to(device)
            fake = torch.zeros(batch_size, dtype=torch.long).to(device)
            """
            train generator
            """
            optimizer_g.zero_grad()
            z=random_z(batch_size,z_dim).to(device)
            gen_conditions=random_labels(batch_size,c_dim).to(device)
            gen_imgs=g_model(z,gen_conditions)
            predicts=d_model(gen_imgs,gen_conditions)
            loss_g=Criterion(predicts,real)

            loss_g.backward(retain_graph=True)
            optimizer_g.step()

            """
            train discriminator
            """
            optimizer_d.zero_grad()
            # loss for real images
            predicts=d_model(images,conditions)
            loss_real=Criterion(predicts,real)
            # loss for fake images
            predicts=d_model(gen_imgs,gen_conditions)
            loss_fake=Criterion(predicts,fake)
            loss_d=(loss_real+loss_fake)/2
            loss_d.backward(retain_graph=True)
            optimizer_d.step()

            print(f'epoch{epoch} {i}/{len(dataloader)}  loss_g: {loss_g.item():.3f}  loss_d: {loss_d.item():.3f}')
            total_loss_g+=loss_g.item()
            total_loss_d+=loss_d.item()

        # evaluate
        z=random_z(len(test_conditions),z_dim).to(device)
        gen_imgs=g_model(z,test_conditions)
        score=evaluation_model.eval(gen_imgs,test_conditions)
        if score>best_score:
            best_score=score
            best_model_wts=copy.deepcopy(g_model.state_dict())
            torch.save(best_model_wts,f'score{score:.2f}.pt')
        print(f'avg loss_g: {total_loss_g/len(dataloader):.3f}  avg_loss_d: {total_loss_d/len(dataloader):.3f}')
        print(f'testing score: {score:.2f}')
        print('---------------------------------------------')

def random_z(batch_size,z_dim):
    return torch.randn(batch_size,z_dim)

def random_labels(batch_size,c_dim):
    pick_num=np.random.randint(1,4)
    pick=np.random.choice(24,pick_num,replace=False)
    labels=torch.zeros(batch_size,c_dim)
    for i in pick:
        labels[i]=1.
    return labels






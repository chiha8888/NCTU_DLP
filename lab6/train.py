import torch
import numpy as np
import copy
import os

from util import get_test_conditions,save_image
from evaluator import EvaluationModel
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(dataloader,g_model,d_model,z_dim,epochs,lr,LAMBDA,n_critic):
    """
    :param z_dim: 100
    """
    optimizer_g=torch.optim.Adam(g_model.parameters(),lr,betas=(0.5,0.99))
    optimizer_d=torch.optim.Adam(d_model.parameters(),lr,betas=(0.5,0.99))
    evaluation_model=EvaluationModel()

    test_conditions=get_test_conditions().to(device)
    fixed_z = random_z(len(test_conditions), z_dim).to(device)
    best_score = 0

    for epoch in range(1,1+epochs):
        total_loss_g=0
        total_loss_d=0
        for i,(images,conditions) in enumerate(dataloader):
            g_model.train()
            d_model.train()
            batch_size=len(images)
            images = images.to(device)
            conditions = conditions.to(device)

            """
            train discriminator
            """
            optimizer_d.zero_grad()

            # for real images
            real_predicts = d_model(images, conditions)
            # for fake images
            z = random_z(batch_size, z_dim).to(device)
            gen_imgs = g_model(z,conditions)
            fake_predicts = d_model(gen_imgs.detach(), conditions)
            # for gradient penalty
            gradient_penalty=calc_gradient_penalty(d_model,images,gen_imgs,conditions)

            # bp
            # WGAN-GP define real=-1 , fake=+1
            loss_d = -torch.mean(real_predicts)+torch.mean(fake_predicts)+LAMBDA*gradient_penalty
            loss_d.backward()
            optimizer_d.step()

            """
            train generator every n_critic iterations
            """
            if i%n_critic==0:
                optimizer_g.zero_grad()

                z = random_z(batch_size, z_dim).to(device)
                gen_imgs = g_model(z, conditions)
                fake_predicts = d_model(gen_imgs,conditions)

                # bp
                loss_g = -torch.mean(fake_predicts)
                loss_g.backward()
                optimizer_g.step()
                print(f'epoch{epoch} {i}/{len(dataloader)}  loss_g: {loss_g.item():.3f}  loss_d: {loss_d.item():.3f}')
                total_loss_d+=loss_d.item()
                total_loss_g+=loss_g.item()

        # evaluate
        g_model.eval()
        d_model.eval()
        with torch.no_grad():
            gen_imgs=g_model(fixed_z,test_conditions)
        score=evaluation_model.eval(gen_imgs,test_conditions)
        if score>best_score:
            best_score=score
            best_model_wts=copy.deepcopy(g_model.state_dict())
            torch.save(best_model_wts,os.path.join('models',f'epoch{epoch}_score{score:.2f}.pt'))
        print(f'avg loss_g: {total_loss_g/(len(dataloader)//n_critic+1):.3f}  avg_loss_d: {total_loss_d/(len(dataloader)//n_critic+1):.3f}')
        print(f'testing score: {score:.2f}')
        print('---------------------------------------------')
        # savefig
        save_image(gen_imgs, os.path.join('results',f'epoch{epoch}.png'), nrow=8, normalize=True)

def calc_gradient_penalty(netD, real_data, fake_data, conditions):
    # print "real_data: ", real_data.size(), fake_data.size()
    batch_size = real_data.shape[0]
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand(real_data.shape).to(device)
    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)

    disc_interpolates = netD(interpolates,conditions)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def random_z(batch_size,z_dim):
    return torch.randn(batch_size,z_dim)

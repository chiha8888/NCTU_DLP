from __future__ import print_function
import os
import sys
from glob import glob
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import evaluator
from evaluator import EvaluationModel as eval_model

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Number of workers for dataloader
workers = 2
# Batch size during training
batch_size = 64
# Spatial size of training images. All images will be resized to this
# size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 100
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0)
                      else "cpu")

    
def get_onehot_label(values, obj_dict):
    onehot = np.zeros(24)
    for value in values:
        class_id = obj_dict[value]
        onehot[class_id] = 1.
    
    return onehot


def get_data():
    print("Start reading train file.")
    TRAIN_FILE_PATH = "dataset/train.json"
    OBJECTS_FILE_PATH = "dataset/objects.json"
    
    # Read the training data from train.json
    train_data = {}
    with open(TRAIN_FILE_PATH, "r") as train_read_file:
        train_data = json.load(train_read_file)
    
    # Read the object dict from objects.json
    obj_dict = {}
    with open(OBJECTS_FILE_PATH, "r") as obj_read_file:
        obj_dict = json.load(obj_read_file)
    
    # Store the image name and one-hot label
    img_names = []
    labels = []
    for key, values in train_data.items():
        img_names.append(key)
        # Get one hot label from image (a np array)
        labels.append(get_onehot_label(values, obj_dict))
    
    return img_names, labels


class CGAN_loader(data.Dataset):
    def __init__(self, root):
        """
        Args:
            root (string): Root path of the dataset.

            self.img_names (string list): String list that store all image names.
            self.labels (int or float list): Numerical list that store all ground truth label values(onehot vector).
        """
        self.root = root
        self.img_names, self.labels = get_data()
        self.data_transforms = \
            transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                   (0.5, 0.5, 0.5)),
                            ])
        
        print("> Found %d images..." % (len(self.img_names)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_names)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_names' and load it.
           
           step2. Get the ground truth label from self.labels
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        # step1.
        path = self.root + self.img_names[index]
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # step2.
        label = self.labels[index]

        # step3. Use the transform.ToTensor() can accomplish two tasks in hint
        # Can also apply more transform on the image.
        img = self.data_transforms(img)

        return img, label


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        
        self.ylabel =nn.Sequential(
            nn.Linear(24, 300),
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + 300, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, onehot_label):
        linear_label = self.ylabel(onehot_label)
        x = torch.cat([input, linear_label], 1)
        x = x.view(-1, nz + 300, 1, 1)
        return self.main(x)


# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
# to mean=0, stdev=0.2.
netG.apply(weights_init)


# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()

        self.ngpu = ngpu

        self.ylabel = nn.Sequential(
            nn.Linear(24, 64*64*1),
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, onehot_label):
        linear_label = self.ylabel(onehot_label)
        linear_label = linear_label.view(-1, 1, 64, 64)
        x = torch.cat([input, linear_label], 1)
        return self.main(x)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
# to mean=0, stdev=0.2.
netD.apply(weights_init)


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Load data
IMAGE_ROOT_PATH = "iclevr/"
train_data = CGAN_loader(IMAGE_ROOT_PATH)
dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)


# Directory to save trainning weight
save_weight_dir = 'eval_images'

# Lists to keep track of progress
G_losses = []
D_losses = []

# Training Loop
def train_GAN(current_epoch_num):
    print("Starting Training Loop...")
    outputD_real = 0.0
    outputD_noise = 0.0
    outputG = 0.0
    count_batch = 0
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            onehot_label = data[1].to(device)
            onehot_label = onehot_label.float()
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            outputD_real = netD(real_cpu, onehot_label).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(outputD_real, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, device=device)
            # Generate fake image batch with G
            fake = netG(noise, onehot_label)
            label.fill_(fake_label)
            # Classify all fake batch with D
            outputD_noise = netD(fake.detach(), onehot_label).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(outputD_noise, label)
            # Calculate the gradients for this batch
            errD_fake.backward()

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of
            # all-fake batch through D
            outputG = netD(fake, onehot_label).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(outputG, label)
            # Calculate gradients for G
            errG.backward()

            # Update G
            optimizerG.step()

            batches_done = epoch * len(dataloader) + i
            
            # Save the images for checking generate result
            sample_images = fake

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item())
            )

        # Output training stats every epoch
        """ D_x = outputD_real.mean().item()
        D_G_z1 = outputD_noise.mean().item()
        D_G_z2 = outputG.mean().item() """

        """ print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)) """
        
        # Save the images for checking generate result
        save_image(sample_images.data[:9], "output_images/%d.png" % epoch, nrow=3, normalize=True)
             
        if epoch % 50 == 0:
            # Do checkpointing for every epoch
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth'
                       % (save_weight_dir, epoch + current_epoch_num))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth'
                       % (save_weight_dir, epoch + current_epoch_num))


# For evaluation
def get_test_onehot():
    # Read test labels
    TEST_FILE_PATH = "dataset/test.json"
    with open(TEST_FILE_PATH, "r") as test_read_file:
        test_label = json.load(test_read_file)

    # Read the object dict from objects.json
    OBJECTS_FILE_PATH = "dataset/objects.json"
    obj_dict = {}
    with open(OBJECTS_FILE_PATH, "r") as obj_read_file:
        obj_dict = json.load(obj_read_file)
    
    labels = []
    for values in test_label:
        labels.append(get_onehot_label(values, obj_dict))

    return labels


def eval_GAN():
    with torch.no_grad():
        model_name = "netG_epoch_199"
        EVAL_PATH = model_name + ".pth"
        eval_G = Generator(ngpu).to(device)
        eval_G = eval_G.eval()
        eval_G.apply(weights_init)
        eval_G.load_state_dict(torch.load(EVAL_PATH))
    
        z_vetcors = torch.randn(32, nz, device=device)
        test_onehot_label = torch.FloatTensor(get_test_onehot()).to(device)
        generate_images = eval_G(z_vetcors, test_onehot_label).detach()

        # Dimension of generate images [32, 3, 64, 64]
        # Dimension of onehot label [32, 24]
        test_model = eval_model()
        accuracy = test_model.eval(generate_images, test_onehot_label)
        print("Accuracy: ", accuracy)
        
        generate_images = generate_images.cpu()
        # Save the images for checking generate result
        save_image(generate_images, "eval_images/" + model_name + ".png", nrow=8, normalize=True)
    
       


if __name__ == '__main__':
    train_GAN(0)

    # Plot the images for loss
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png")

    eval_GAN()


        

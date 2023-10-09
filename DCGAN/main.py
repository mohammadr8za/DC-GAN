import os

import matplotlib.pyplot as plt
import torch
from models import models
from torch import nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from data_preparation import ImageDatasetGAN
import matplotlib.pyplot as plt
from engine import run
from torchvision.datasets import MNIST
from os.path import join

# I N I T I A L I Z A T I O N
root = r"D:\mreza\TestProjects\Python\DCGAN"
data_file = "annotation.csv"
dataset_id = ["Cars4GAN"]

Epochs = 100
z_dim = [100]
learning_rate = [0.0002]
batch_size = 128

# device diagnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# define manual transform
manual_transform = transforms.Compose([transforms.Resize((64, 64)),
                                       transforms.PILToTensor(),
                                       transforms.RandomRotation(0.5)])

# defining dataset
car_dataset = ImageDatasetGAN(root=root, data_file="annotation.csv", dataset_id="Cars4GAN", manual_transform=manual_transform)
img_sample, label_sample = car_dataset[0]
img_sample.shape, label_sample

mnist_root = join(root, "Data", "MNIST")
os.makedirs(mnist_root, exist_ok=True)
mnist_dataset = MNIST(root=mnist_root, transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Resize((64, 64))]),
                      download=True)

# defining dataloader
car_dataloader = DataLoader(dataset=car_dataset, batch_size=batch_size)
imgs, labels = next(iter(car_dataloader))
imgs.shape, labels.shape

mnist_dataloader = DataLoader(dataset=mnist_dataset, batch_size=batch_size)

config = {"dataset": None,
          "lr": None,
          "z_dim": None}

for dataset in dataset_id:
    config["dataset"]=dataset

    for lr in learning_rate:
        config["lr"] = lr

        for dim in z_dim:
            config["z_dim"] = dim

            Exp_ID = fr"id{config['dataset']}__zdim{config['z_dim']}__lr{config['lr']}"

            # define models
            gen_net = models.GeneratorNetwork(z_dim=config['z_dim'], image_channels=1).to(device)
            dis_net = models.DiscriminatorNetwork(image_channels=1).to(device)

            # define criterion and optimizer
            loss_fn = nn.BCELoss()
            optimizer_dis = torch.optim.Adam(params=dis_net.parameters(), lr=lr/250, betas=(0.5, 0.999))
            optimizer_gen = torch.optim.Adam(params=gen_net.parameters(), lr=lr, betas=(0.5, 0.999))

            run(gen_net=gen_net,
                dis_net=dis_net,
                dataloader=mnist_dataloader,
                device=device,
                loss_fn=loss_fn,
                optimizer_gen=optimizer_gen,
                optimizer_dis=optimizer_dis,
                Exp_ID=Exp_ID,
                epochs=Epochs,
                config=config,
                root=root)

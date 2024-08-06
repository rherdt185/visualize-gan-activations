import torch
from torchvision.models import resnet50

import pickle
import random

import numpy as np


def load_model(model_str):
    if model_str == "resnet50":
        return resnet50(pretrained=True).to("cuda").eval()



def load_gan(gan_model="afhqwild.pkl"):
    device = torch.device('cuda')
    with open(gan_model, 'rb') as f:

        data = pickle.load(f)
        G = data['G_ema'].to(device).eval()
        #G = legacy.load_network_pkl("model_ckpts/stylegan2/brecahad.pkl")['G_ema'].to(device).eval()
    #self.network_gan = WrapperStyleGAN2(G)

    label = torch.zeros([1, G.c_dim], device="cuda")
    z = torch.from_numpy(np.random.RandomState(random.randint(0, 100000000)).randn(1, G.z_dim)).to("cuda")

    return G
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
import numpy as np
import pandas as pd
import random
import os
import time
from IPython import display
from dataclasses import dataclass
import sys

import torch
from torch import nn

import torchaudio

import librosa
from matplotlib import pyplot as plt

from src.vocoder import Vocoder

vocoder = Vocoder()
vocoder = vocoder.eval()

from src.melspectrogram import MelSpectrogram, MelSpectrogramConfig

featurizer = MelSpectrogram(MelSpectrogramConfig())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
BATCH_SIZE = 8

torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


from src.dataset import load_dataset
dataloader_train, dataloader_val = load_dataset(featurizer, BATCH_SIZE)

from model import Tacotron2

generator = Tacotron2(n_mels=80, n_frames=1).to(device)

from math import exp, log
optimizer = optim.Adam(generator.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-6, weight_decay=1e-6)
lambda1 = lambda step: exp(log(0.01)*min(15000, step)/15000)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import wandb
wandb.init(
  project="DLA_HW4",
  config={
    "lstm_dropout": 0.1,
    "n_mels": 80,
    "n_frames": 1,
    "learn_rate": 0.001,
    "Guided attention": 1,
    "batch_size": 64,
    "epochs": 80 }
)
config = wandb.config
api = wandb.Api()

def initialize_attn_matrix(B, batch_text_size, batch_audio_size, text_length, audio_length):
    g = 0.2
    res = None
    for i in range(B):
        T = text_length[i].item()
        N = audio_length[i].item()
        one_image = torch.arange(N).view(N, 1).repeat(1, T).float()/N
        one_image -= torch.arange(T).view(1, T).repeat(N, 1).float()/T
        one_image = F.pad(one_image, (0, batch_audio_size - T, 0, batch_text_size - N))
        one_image = one_image.unsqueeze(0)
        if i == 0:
            res = one_image
        else:
            res = torch.cat([res, one_image], dim=0)
    res = 1 - torch.exp(-res**2/2*g*g)
    return res.view(B, batch_text_size, batch_audio_size)

criterion = nn.MSELoss()
crossentropy = nn.BCELoss()
def run_training(epochs):
    for epoch in range(epochs):
        generator.train()
        train_losses = []
        attn_losses = []
        i = 0
        acc = 0
        print('Epoch', epoch)
        for audio_b, audio_length_b, text_b, text_length_b in dataloader_train:
            if i%20 == 0:
                for param_group in optimizer.param_groups:
                    print(param_group['lr'])
            audio_b, text_b = audio_b.to(device), text_b.to(device)
            audio_length_b, text_length_b = audio_length_b.to(device), text_length_b.to(device)
            pad_mask = (text_b != 0).to(device)
            res, before_prenet, stop_token, attn_matrix = generator(text_b, pad_mask, audio_b, device)
            audio_pad_mask = (audio_b != 0).type(torch.float)
            loss = criterion(audio_b, res*audio_pad_mask)
            loss += criterion(audio_b, before_prenet*audio_pad_mask)
            
            audio_pad_mask = audio_pad_mask[:, 0, :].squeeze(1)
            stop_token = stop_token.squeeze()
            true_stop = torch.roll((1 - audio_pad_mask), -1, 1)
            true_stop[:, -1] = torch.ones(true_stop.shape[0])
            loss_stop = crossentropy(stop_token.type(torch.float), true_stop.type(torch.float))
            loss += loss_stop

            attn_masked = (attn_matrix*audio_pad_mask.unsqueeze(1).repeat(1, attn_matrix.shape[1], 1))
            attn_masked = (attn_matrix*pad_mask.unsqueeze(2).repeat(1, 1, attn_matrix.shape[2]))
            perfect_attn = initialize_attn_matrix(attn_matrix.shape[0], attn_matrix.shape[1], attn_matrix.shape[2], audio_length_b, text_length_b).to(device)
            loss_attention = torch.sum(attn_masked*perfect_attn)/attn_matrix.shape[0]
            loss += loss_attention

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())
            attn_losses.append(loss_attention.item())
            if i % 20 == 0:
                print(i, np.mean(train_losses[-20:]), np.mean(attn_losses[-20:]))
            i += 1
        print('Train loss', np.mean(train_losses))
        generator.eval()
        val_losses = []
        with torch.no_grad():
            i = 0
            for audio_b, audio_length_b, text_b, text_length_b in dataloader_val:
                audio_b, text_b = audio_b.to(device), text_b.to(device)
                audio_length_b, text_length_b = audio_length_b.to(device), text_length_b.to(device)
                pad_mask = (text_b != 0).to(device)
                res, before_prenet, stop_token, attn_matrix = generator(text_b, pad_mask, audio_b, device)
                audio_pad_mask = (audio_b != 0).type(torch.float)
                res_masked = res*audio_pad_mask
                loss = criterion(audio_b, (res*audio_pad_mask))
                loss += criterion(audio_b, (before_prenet*audio_pad_mask))
                
                audio_pad_mask = audio_pad_mask[:, 0, :].squeeze(1)
                stop_token = stop_token.squeeze()
                true_stop = torch.roll((1 - audio_pad_mask), -1, 1)
                true_stop[:, -1] = torch.ones(true_stop.shape[0])
                true_stop *= audio_pad_mask
                loss_stop = crossentropy(stop_token*audio_pad_mask.type(torch.float), true_stop.type(torch.float))
                loss += loss_stop
                
                val_losses.append(loss.item())
                if i < 8:
                    name = "example " + str(i)
                    wandb_gen = wandb.Image(res[0, :, :audio_length_b[0].item()].detach().cpu().numpy(), caption="Generated")
                    wandb_real = wandb.Image(audio_b[0, :, :audio_length_b[0].item()].detach().cpu().numpy(), caption="Real")
                    wandb_attn = wandb.Image(256*attn_matrix[0, :text_length_b[0].item(), :audio_length_b[0].item()].detach().cpu().numpy(), caption="Attention")
                    wandb_images = [wandb_gen, wandb_real, wandb_attn]
                    wandb.log({name: wandb_images}, step=epoch)

                i += 1
                if i < 5:
                    audio_gen = vocoder.inference(res[:1, :, :audio_length_b[0].item()].detach().cpu())
                    audio_real = vocoder.inference(audio_b[:1, :, :audio_length_b[0].item()].detach().cpu())
                    torchaudio.save("temp_gen"+str(i)+".wav", audio_gen, sample_rate=22050)
                    torchaudio.save("temp_real"+str(i)+".wav", audio_real, sample_rate=22050)
                    name = "audio " + str(i)
                    wandb.log({name: [wandb.Audio("temp_gen"+str(i)+".wav", caption="Generated", sample_rate=22050), wandb.Audio("temp_real"+str(i)+".wav", caption="Real", sample_rate=22050)]}, step=epoch)
                    api.flush()
        print('Val loss', np.mean(val_losses))
        to_save = {'weights': generator.state_dict(), 'optimizer': optimizer.state_dict()}
        name = './epoch' + str(epoch)
        torch.save(to_save, name)
        wandb.log({"Train loss": np.mean(train_losses), "Val loss": np.mean(val_losses), "Attention loss": np.mean(attn_losses)}, step=epoch)
        
epochs = int(sys.argv[1])
run_training(epochs)
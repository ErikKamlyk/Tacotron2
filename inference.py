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

torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from model import Tacotron2

generator = Tacotron2(n_mels=80, n_frames=1).to(device)

from math import exp, log
optimizer = optim.Adam(generator.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-6, weight_decay=1e-6)
lambda1 = lambda step: exp(log(0.01)*min(15000, step)/15000)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])

full_model = torch.load('model_final')
generator.load_state_dict(full_model['weights'])
optimizer.load_state_dict(full_model['optimizer'])

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

def run_inference(text, audio=None):
        generator.eval()
        text = [ord(c) for c in text if ord(c) < 256]
        text = torch.tensor(text).view(1, -1)
        with torch.no_grad():
            text = text.to(device)
            pad_mask = (text != 0).to(device)
            res, before_prenet, stop_token, attn_matrix = generator(text, pad_mask, None, device)
            wandb_gen = wandb.Image(res[0, :, :].detach().cpu().numpy(), caption="Generated")
            wandb_attn = wandb.Image(256*attn_matrix[0, :, :].detach().cpu().numpy(), caption="Attention")
            wandb_images = [wandb_gen, wandb_attn]
            audio_gen = vocoder.inference(res[:1, :, :].detach().cpu())
            torchaudio.save("gen.wav", audio_gen, sample_rate=22050)
            wandb_audios = [wandb.Audio("gen.wav", caption="Generated", sample_rate=22050)]
            if audio != None:
                wandb_real = wandb.Image(audio[0, :, :].detach().cpu().numpy(), caption="Real")
                wandb_images.append(wandb_real)
                audio_real = vocoder.inference(audio[:1, :, :])
                torchaudio.save("temp_real.wav", audio_real, sample_rate=22050)
                wandb_audios.append(wandb.Audio("temp_real.wav", caption="Real", sample_rate=22050))
            wandb.log({"mels": wandb_images}, step=0)
            wandb.log({"audios": wandb_audios}, step=0)
            api.flush()
        
text = sys.argv[1]
run_inference(text)
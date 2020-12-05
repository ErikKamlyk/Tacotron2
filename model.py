from LSAttention import LocationSensitiveAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
import numpy as np

class Prenet(nn.Module):
    def __init__(self, n_mels, n_frames):
        super(Prenet, self).__init__()
        self.fc1 = nn.Linear(n_mels*n_frames, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, melspec):
        x = F.relu(self.dropout1(self.fc1(melspec)))
        x = F.relu(self.dropout2(self.fc2(x)))
        return x

class Postnet(nn.Module):
    def __init__(self, n_mels):
        super(Postnet, self).__init__()
        self.conv1 = nn.Conv1d(n_mels, 512, 5, padding=2)
        self.bn_conv1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 512, 5, padding=2)
        self.bn_conv2 = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv1d(512, 512, 5, padding=2)
        self.bn_conv3 = nn.BatchNorm1d(512)
        self.conv4 = nn.Conv1d(512, 512, 5, padding=2)
        self.bn_conv4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, n_mels, 5, padding=2)
        self.bn_conv5 = nn.BatchNorm1d(n_mels)

    def forward(self, x):
        x = F.tanh(self.bn_conv1(self.conv1(x)))
        x = F.tanh(self.bn_conv2(self.conv2(x)))
        x = F.tanh(self.bn_conv3(self.conv3(x)))
        x = F.tanh(self.bn_conv4(self.conv4(x)))
        return self.bn_conv5(self.conv5(x))

class Tacotron2(nn.Module):
    def __init__(self, n_mels, n_frames):
        super(Tacotron2, self).__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.embed = nn.Embedding(256, 512)

        self.conv_en1 = nn.Conv1d(512, 512, 5, padding=2)
        self.bn_conv_en1 = nn.BatchNorm1d(512)
        self.conv_en2 = nn.Conv1d(512, 512, 5, padding=2)
        self.bn_conv_en2 = nn.BatchNorm1d(512)
        self.conv_en3 = nn.Conv1d(512, 512, 5, padding=2)
        self.bn_conv_en3 = nn.BatchNorm1d(512)

        self.bilstm = nn.LSTM(512, 256, bidirectional=True, batch_first=True, dropout=0.1)
        self.LSA = LocationSensitiveAttention(1024, 512, 128, 32, 31)

        self.decoder = nn.LSTM(256+512, 1024, 2, dropout=0.1)

        self.get_mels = nn.Linear(1024+512, n_mels*n_frames)
        self.stop_layer = nn.Linear(1024+512, n_frames)

        self.prenet = Prenet(n_mels, n_frames)

        self.postnet = Postnet(n_mels)
        
    def forward(self, x, mask, target, device='cpu'):
        x = self.embed(x).permute(0, 2, 1)

        x = F.relu(self.bn_conv_en1(self.conv_en1(x)))
        x = F.relu(self.bn_conv_en2(self.conv_en2(x)))
        x = F.relu(self.bn_conv_en3(self.conv_en3(x)))

        x = self.bilstm(x.permute(0, 2, 1))[0]
        cum_attn_weights = torch.zeros(x.shape[0], x.shape[1]).to(device)
        attn_weights = torch.zeros(x.shape[0], x.shape[1]).to(device)
        attn_weights_cat = torch.cat([cum_attn_weights.unsqueeze(1), attn_weights.unsqueeze(1)], dim=1).to(device)
        prev = torch.zeros((x.shape[0], 1, 256)).to(device)
        decoded = torch.zeros((x.shape[0], 1, 1024)).to(device)
        h = None
        melspec = None
        stop_vector = None
        i = 0
        attention_matrix = None
        stop = False
        while not stop:
            context, attn_weights = self.LSA(decoded, x, attn_weights_cat, ~mask)
            frames_to_gen = self.n_frames
            if target != None:
                frames_to_gen = min(self.n_frames, target.shape[-1] - i)
            if attention_matrix == None:
                attention_matrix = attn_weights.unsqueeze(2).repeat(1, 1, frames_to_gen)
            else:
                attention_matrix = torch.cat([attention_matrix, attn_weights.unsqueeze(2).repeat(1, 1, frames_to_gen)], dim=2)
            cum_attn_weights += attn_weights
            attn_weights_cat = torch.cat([cum_attn_weights.unsqueeze(1), attn_weights.unsqueeze(1)], dim=1).to(device)
            if h != None:
                decoded, h = self.decoder(torch.cat([prev, context], dim=2), h)
            else:
                decoded, h = self.decoder(torch.cat([prev, context], dim=2))
            mels = self.get_mels(torch.cat([decoded, context], dim=2)).permute(0, 2, 1).view(mask.shape[0], self.n_mels, self.n_frames)
            stop_prob = F.sigmoid(self.stop_layer(torch.cat([decoded, context], dim=2)))
            
            if target != None:
                target_frames = target[:, :, i:i+self.n_frames]
                stop_prob = stop_prob[:, :, :target_frames.shape[-1]].squeeze(1)
                if target_frames.shape[-1] == self.n_frames:
                    prev = self.prenet(target_frames.reshape(mask.shape[0], self.n_mels*self.n_frames))
                    prev = prev.unsqueeze(1)
                else:
                    mels = mels[:, :, :target_frames.shape[-1]]
            else:
                stop_prob = stop_prob.squeeze(1)
                prev = self.prenet(mels.reshape(mask.shape[0], self.n_mels*self.n_frames))
                prev = prev.unsqueeze(1)
            
            i += self.n_frames
            if target != None and i >= target.shape[2]:
                stop = True
            elif target == None:
                if i > 2000:
                    stop = True
                for j in range(self.n_frames):
                    if i > mask.shape[1] and stop_prob[0, j] >= 0.5:
                        stop = True
                        mels = mels[:, :, :j+1]
                        stop_prob = stop_prob[:, :j+1]
   
            if melspec != None:
                melspec = torch.cat([melspec, mels], dim=2)
            else:
                melspec = mels
            if stop_vector != None:
                stop_vector = torch.cat([stop_vector, stop_prob], dim=1)
            else:
                stop_vector = stop_prob
        final_melspec = melspec + self.postnet(melspec)
        #attn_weights, attn_weights_cat, cum_attn_weights, decoded, prev = attn_weights.detach().cpu(), attn_weights_cat.detach().cpu(), cum_attn_weights.detach().cpu(), decoded.detach().cpu(), prev.detach().cpu()
        return final_melspec, melspec, stop_vector, attention_matrix
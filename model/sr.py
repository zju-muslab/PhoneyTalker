import torch
import torch.nn as nn
import torchaudio.functional as func
from model.modules import *


class Dvector(nn.Module):
    
    def __init__(self, n_mfcc, n_speaker, p_dropout, pool_size):
        super(Dvector, self).__init__()
        self.n_mfcc = n_mfcc
        self.n_speaker = n_speaker
        self.p_dropout = p_dropout
        self.pool_size = pool_size
        self.encoder = nn.Sequential(
            Maxout(n_mfcc, 512, 0, pool_size),
            Maxout(512, 512, 0, pool_size),
            Maxout(512, 512, p_dropout, pool_size),
            Maxout(512, 512, p_dropout, pool_size)
        )
        self.fc = nn.Linear(512, n_speaker)
        
    def forward(self, x):
        frame_emb = self.encoder(x)
        hidden_emb = frame_emb.mean(dim=2)
        return self.fc(hidden_emb) if self.training else [hidden_emb, hidden_emb]
    
    
class DeepSpeaker(nn.Module):
    
    def __init__(self, n_mfcc, n_speaker, p_dropout):
        super(DeepSpeaker, self).__init__()
        self.n_mfcc = n_mfcc
        self.n_speaker = n_speaker
        self.p_dropout = p_dropout
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(1, 64, 3)
        self.layer2 = self._make_layer(64, 128, 3)
        self.layer3 = self._make_layer(128, 256, 3)
        self.layer4 = self._make_layer(256, 512, 3)
        self.segment = Segment(512, 512, p_dropout)
        self.fc = nn.Linear(512, n_speaker)
        
    def _make_layer(self, in_planes, out_planes, n_block):
        layers = [
            nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=2),
            nn.BatchNorm2d(out_planes),
            self.relu
        ]
        for _ in range(n_block):
            layers.append(ResBlock(out_planes, out_planes))
        return nn.Sequential(*layers)
    
    @classmethod
    def cat_delta(cls, x):
        delta1 = func.compute_deltas(x)
        delta2 = func.compute_deltas(delta1)
        return torch.cat((x, delta1, delta2), dim=1)
        
    def forward(self, x):
        x = self.cat_delta(x)
        x = x.unsqueeze(dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        frame_emb = self.layer4(x)
        hidden_emb = torch.mean(frame_emb, dim=-1).squeeze(dim=-1).squeeze(dim=-1)
        segment_emb = self.segment(hidden_emb)
        return self.fc(segment_emb) if self.training else [hidden_emb, segment_emb]
    

class Xvector(nn.Module):

    def __init__(self, n_mfcc, n_speaker, p_dropout=0.2):
        super(Xvector, self).__init__()
        self.n_mfcc = n_mfcc
        self.n_speaker = n_speaker
        self.p_dropout = p_dropout
        self.frame_encoder = nn.Sequential(
            Frame(n_mfcc, 512, 5, 1, p_dropout),
            Frame(512, 512, 5, 2, p_dropout),
            Frame(512, 512, 7, 3, p_dropout),
            Frame(512, 512, 1, 1, p_dropout),
            Frame(512, 1500, 1, 1, p_dropout)
        )
        self.segment_encoder1 = Segment(3000, 512, p_dropout)
        self.segment_encoder2 = Segment(512, 512, p_dropout)
        self.fc = nn.Linear(512, n_speaker)
    
    def forward(self, x):
        frame_emb = self.frame_encoder(x)
        hidden_emb = torch.cat((frame_emb.mean(dim=2), frame_emb.std(dim=2)), dim=1)
        segment_emb = self.segment_encoder1(hidden_emb)
        return self.fc(self.segment_encoder2(segment_emb)) if self.training else [hidden_emb, segment_emb]
    
    

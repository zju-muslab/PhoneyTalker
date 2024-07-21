import torch
import torch.nn as nn
import numpy as np
from model.modules import MFCC
from model.gplda import GPLDA
from model.sr import Dvector, Xvector, DeepSpeaker


def load_encoder(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if ckpt_path.find('xvector') >= 0:
        encoder = Xvector(n_mfcc=ckpt['n_mfcc'], n_speaker=ckpt['n_speaker'], p_dropout=ckpt['p_dropout'])
    elif ckpt_path.find('dvector') >= 0:
        encoder = Dvector(n_mfcc=ckpt['n_mfcc'], n_speaker=ckpt['n_speaker'], p_dropout=ckpt['p_dropout'], pool_size=ckpt['pool_size'])
    elif ckpt_path.find('deepspeaker') >= 0:
        encoder = DeepSpeaker(n_mfcc=ckpt['n_mfcc'], n_speaker=ckpt['n_speaker'], p_dropout=ckpt['p_dropout'])
    else:
        raise Exception('Invalid ckpt path!')
    encoder.load_state_dict(ckpt['encoder'])
    return encoder


def load_gplda(model_path):
    arrs = np.load(model_path)
    return GPLDA(arrs['phi'], arrs['sigma'], arrs['W'], arrs['miu'])


class Discriminator(nn.Module):

    def __init__(self, mfcc_cfg, encoder_path, gplda_path, threshold, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.mfcc = MFCC(
            sample_rate=mfcc_cfg.sample_rate,
            win_length=mfcc_cfg.win_len_ms * mfcc_cfg.sample_rate // 1000,
            hop_length=mfcc_cfg.hop_len_ms * mfcc_cfg.sample_rate // 1000,
            n_fft=mfcc_cfg.n_fft,
            n_mfcc=mfcc_cfg.n_mfcc,
            cmn_window=mfcc_cfg.cmn_window
        )
        self.mfcc.to(device)
        self.encoder = load_encoder(encoder_path)
        self.encoder.to(device)
        self.scorer = load_gplda(gplda_path)
        self.scorer.to_device(device)
        self.threshold = threshold

    def enroll(self, dataloader):
        enrolled_profiles = [[] for _ in range(len(dataloader.dataset.speakers))]
        with torch.no_grad():
            for _, _, data, label in dataloader:
                emb = self.encode(data.to(self.device))[-1]
                enrolled_profiles[label.item()].append(emb.detach())
        enrolled_embs = [torch.vstack(embs).mean(dim=0) for embs in enrolled_profiles]
        self.enrolled_embs = torch.vstack(enrolled_embs)
        
    def encode(self, x):
        self.encoder.eval()
        return self.encoder(self.mfcc(x.squeeze(dim=1)))
    
    def score(self, test_emb):
        return self.scorer(self.enrolled_embs, test_emb).T
    
    def forward(self, x):
        test_emb = self.encode(x)[-1]
        return self.score(test_emb)

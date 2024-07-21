import random
import torch
import torch.nn as nn
import torchaudio as ta


class Generator(nn.Module):
    
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.cfg = cfg
        self.phonemes = nn.Parameter(torch.zeros((self.cfg.phoneme_num, self.cfg.phoneme_len)))
        nn.init.xavier_normal_(self.phonemes)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.cfg.phoneme_len, self.cfg.phoneme_len * self.cfg.scale_factor),
            nn.ReLU(),
            nn.Linear(self.cfg.phoneme_len * self.cfg.scale_factor, self.cfg.phoneme_len * self.cfg.scale_factor),
            nn.ReLU(),
            nn.Linear(self.cfg.phoneme_len * self.cfg.scale_factor, self.cfg.phoneme_len * self.cfg.scale_factor),
            nn.ReLU(),
            nn.Linear(self.cfg.phoneme_len * self.cfg.scale_factor, self.cfg.phoneme_len),
            nn.Tanh()
        )
    
    def forward(self, data, index_list, offset_list, phonemes_list):
        phoneme_dict = self.mlp(self.phonemes)
        perturb = torch.zeros_like(data)
        for i, j in enumerate(index_list):
            phonemes = phonemes_list[j]
            for start, end, phn_idx in phonemes:
                start = max(start - offset_list[i], 0)
                start = random.randint(start, start + self.cfg.phoneme_len)
                end = min(end - offset_list[i], perturb.shape[-1])
                if start + self.cfg.phoneme_len > end:
                    continue
                num_phonemes = (end - start) // self.cfg.phoneme_len
                end = start + num_phonemes * self.cfg.phoneme_len
                window = torch.hann_window(end - start).to(data.device)
                perturb[i, :, start:end] += torch.hstack([phoneme_dict[phn_idx]] * num_phonemes) * window
        return perturb
    
import torch
import torchaudio as ta


class Suppressor:
    
    def __init__(self, cfg):
        self.cfg = cfg
        
    def __call__(self, perturb):
        perturb = ta.functional.lowpass_biquad(perturb, self.cfg.sample_rate, self.cfg.lp_freq)
        return torch.clip(perturb, -self.cfg.epsilon, self.cfg.epsilon)
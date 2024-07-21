import os
import shutil
import random
import torch
import numpy as np
import torchaudio as ta
from torch.utils.data import DataLoader

from dataset import PhonemeDataset
from model.discriminator import Discriminator


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def set_dir(path, recreate=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif recreate:
        shutil.rmtree(path)
        os.makedirs(path)
        
        
def calc_snr(wav, wav_, epsilon=1e-10):
    power_ratio = torch.sum(wav ** 2, dim=-1)  / (torch.sum((wav_ - wav) ** 2, dim=-1) + epsilon)
    return 10 * torch.log10(power_ratio + epsilon).mean().item()


def save_wave_batch(batch, batch_id, wav_dir):
    set_dir(wav_dir)
    prefix = str(batch_id).zfill(3)
    for i, wav in enumerate(batch):
        ta.save(os.path.join(wav_dir, prefix + '_' + str(i).zfill(3) + '.wav'), wav, 16000)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience, ckpt_path, logger):
        self.patience = patience
        self.ckpt_path = ckpt_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.logger = logger

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            self.logger.warning(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        self.logger.info(f'Saving model at {self.ckpt_path}')
        torch.save({
            'cfg': model.cfg,
            'generator': model.state_dict()
        }, self.ckpt_path)


def load_discriminator(system_name, system_cfg, mfcc_cfg, ckpt_dir, data_dir, cache_dir, device):
    discriminator = Discriminator(mfcc_cfg, 
                                os.path.join(ckpt_dir, system_cfg.encoder_path), 
                                os.path.join(ckpt_dir, system_cfg.gplda_path), 
                                system_cfg.threshold,
                                device).to(device)
    enrolled_emb_path = os.path.join(cache_dir, f'{system_name}_enrolled_embs.pth')
    if os.path.exists(enrolled_emb_path):
        discriminator.enrolled_embs = torch.load(enrolled_emb_path, map_location=device)
    else:
        dataset_enroll = PhonemeDataset(data_dir)
        dataloader_enroll = DataLoader(dataset_enroll)
        discriminator.enroll(dataloader_enroll)
        torch.save(discriminator.enrolled_embs, enrolled_emb_path)
    return discriminator
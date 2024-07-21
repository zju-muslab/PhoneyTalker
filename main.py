import os
import time
import hydra
import logging
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import utils
from dataset import PhonemeDataset
from model.generator import Generator
from model.suppressor import Suppressor


logger = logging.getLogger(__name__)

def run_epoch(cfg, run_mode, dataloader, discriminator, generator, suppressor, optimizer=None):
    device = discriminator.device
    generator.train()
    running_loss, running_asr, running_snr = 0, 0, 0
    for i, batch in enumerate(dataloader):
        index, offset, src_data, _ = batch
        index, offset = index.tolist(), offset.tolist()
        src_data = src_data.to(device)
        
        perturb = generator(src_data, index, offset, dataloader.dataset.phonemes)
        perturb = suppressor(perturb)
        adv_data = torch.clip(src_data + perturb, -1, 1)
        output = discriminator(adv_data)
        if cfg.confidence >= 0:
            loss = torch.mean(torch.clamp_min(discriminator.threshold + cfg.confidence - output[:, cfg.target_speaker], 0))
        else:
            loss = torch.mean(discriminator.threshold - output[:, cfg.target_speaker])
        if run_mode == 'Train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        score = output[:, cfg.target_speaker]
        pred_speaker = output.max(dim=-1)[1]
        res = torch.logical_and(score > discriminator.threshold, pred_speaker == cfg.target_speaker)
        asr = res.float().mean().item()
        src_wav, adv_wav = src_data.detach().cpu(), adv_data.detach().cpu()
        snr = utils.calc_snr(src_wav, adv_wav)
        logger.info(f'{run_mode} >> Batch-[{i:3d}] Loss={loss.item():7.4f} | ASR={asr:6.4f} | SNR={snr:7.4f}')
        running_loss += loss.item() * len(index)
        running_asr += asr * len(index)
        running_snr += snr * len(index)
    n_trials = len(dataloader.dataset)
    return running_loss / n_trials, running_asr / n_trials, running_snr / n_trials
           

def test_epoch(cfg, dataloader, discriminators, generator, suppressor):
    device = discriminators[0].device
    generator.eval()
    n_system = len(discriminators)
    running_asr, running_snr = [0] * n_system, 0
    for i, batch in tqdm(enumerate(dataloader)):
        index, offset, src_data, _ = batch
        index, offset = index.tolist(), offset.tolist()
        src_data = src_data.to(device)

        perturb = generator(src_data, index, offset, dataloader.dataset.phonemes)
        perturb = suppressor(perturb)
        adv_data = torch.clip(src_data + perturb, -1, 1)
        running_snr += utils.calc_snr(src_data.detach().cpu(), adv_data.detach().cpu())
        for k, discriminator in enumerate(discriminators):
            output = discriminator(adv_data)
            score = output[:, cfg.target_speaker]
            pred_speaker = output.max(dim=-1)[1]
            res = torch.logical_and(score > discriminator.threshold, pred_speaker == cfg.target_speaker)
            asr = res.float().mean().item()
            running_asr[k] += asr * len(index)
        if cfg.save_wave:
            utils.save_wave_batch(src_data.detach().cpu(), i, 'wav/src')
            utils.save_wave_batch(adv_data.detach().cpu(), i, 'wav/adv')
    n_trials = len(dataloader.dataset)
    running_asr = torch.tensor(running_asr) / n_trials
    running_snr /= n_trials
    return running_asr.tolist(), running_snr


@hydra.main(version_base=None, config_path='.', config_name='config')
def main(cfg: DictConfig) -> None:
    # configure the environment
    if cfg.gpu:
        device = torch.device(f'cuda:{cfg.device}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    utils.set_seed(cfg.seed)
    ckpt_path = 'generator_ckpt.pth'
    
    if cfg.mode == 'train':
        logger.info('Loading training & validation dataset...')
        dataset_train = PhonemeDataset(os.path.join(cfg.data_dir, 'train'), None, cfg.wav_len)
        dataset_valid = PhonemeDataset(os.path.join(cfg.data_dir, 'valid'), None, cfg.wav_len)
        dataloader_train = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=cfg.batch_size, shuffle=True)
        logger.info(f'Loaded {len(dataset_train)} samples from {len(dataset_train.speakers)} speakers for training!')
        logger.info(f'Loaded {len(dataset_valid)} samples from {len(dataset_valid.speakers)} speakers for validation!')
        logger.info('Loading discriminator...')
        discriminator = utils.load_discriminator(
            cfg.discriminator.name, cfg.discriminator.systems[cfg.discriminator.name], 
            cfg.discriminator.mfcc_cfg, cfg.discriminator.ckpt_dir, 
            os.path.join(cfg.data_dir, 'enroll'), cfg.cache_dir, device)
        discriminator.eval()
        generator = Generator(cfg.generator).to(device)
        suppressor = Suppressor(cfg.suppressor)
        optimizer = torch.optim.Adam(generator.parameters(), cfg.lr)
        stopper = utils.EarlyStopping(cfg.patience, ckpt_path, logger)
        logger.info('Start training and validation...')
        for epoch in range(1, cfg.epoch + 1):
            train_loss, train_asr, train_snr = run_epoch(cfg, 'Train', dataloader_train, discriminator, generator, suppressor, optimizer)
            with torch.no_grad():
                valid_loss, valid_asr, valid_snr = run_epoch(cfg, 'Valid', dataloader_valid, discriminator, generator, suppressor, None)
            logger.info(f'Epoch [{epoch:3d}/{cfg.epoch:3d}] Train# Loss={train_loss:7.4f} | ASR={train_asr:6.4f} | SNR={train_snr:7.4f}')
            logger.info(f'Epoch [{epoch:3d}/{cfg.epoch:3d}] Valid# Loss={valid_loss:7.4f} | ASR={valid_asr:6.4f} | SNR={valid_snr:7.4f}')
            stopper(valid_loss, generator)
            if stopper.early_stop: break
        logger.info('Finished training and validation!')
    else:
        logger.info('Loading testing dataset...')
        dataset_test = PhonemeDataset(os.path.join(cfg.data_dir, 'test'), None)
        dataloader_test = DataLoader(dataset_test, shuffle=False)
        logger.info(f'Loaded {len(dataset_test)} samples from {len(dataset_test.speakers)} speakers for testing!')
        logger.info('Loading discriminator...')
        discriminators = []
        for system_name, system_cfg in cfg.discriminator.systems.items():
            discriminator = utils.load_discriminator(
                system_name, system_cfg, cfg.discriminator.mfcc_cfg, cfg.discriminator.ckpt_dir, 
                os.path.join(cfg.data_dir, 'enroll'), cfg.cache_dir, device)
            discriminator.eval()
            discriminators.append(discriminator)
        logger.info('Loading generator...')
        state_dict = torch.load(ckpt_path, map_location=device)
        generator = Generator(state_dict['cfg']).to(device)
        generator.load_state_dict(state_dict['generator'])
        suppressor = Suppressor(cfg.suppressor)
        logger.info('Start testing...')
        with torch.no_grad():
            test_asr_list, test_snr = test_epoch(cfg, dataloader_test, discriminators, generator, suppressor)
        logger.info(f'[Target on {cfg.discriminator.name}] Test# SNR={test_snr:7.4f}')
        for k, system_name in enumerate(cfg.discriminator.systems.keys()):
            logger.info('-'*100)
            logger.info(f'[{cfg.discriminator.name} -> {system_name:16s}] Test# ASR={test_asr_list[k]:6.4f}')
        logger.info('Finished testing!')


if __name__ == '__main__':
    main()
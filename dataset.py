# %%
import os
import random
import torchaudio
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset


class PhonemeDataset(Dataset):

    """
    Phoneme Code (48)
    ---------------------------------------
    stops             (7): b d g p t k dx [q]
    affricates        (2): jh ch
    fricativaes       (8): s sh z zh f th v dh
    nasals            (4): m_em n_nx ng_eng en
    semivowels&glides (6): l r w y hh_hv el
    vowels           (17): iy ih eh ey ae aa aw ay ah ao oy ow uh uw_ux er_axr ax ix [ax-h]
    others            (4): bcl_dcl_gcl pcl_tcl_kcl_qcl pau_h#_#h epi
    """
    # phoneme code (39)
    phoneme_code = {'AA': 0, 'AE': 1, 'AH': 2, 'AO': 3, 'AW': 4, 'AY': 5, 'B': 6, 'CH': 7, 'D': 8, 'DH': 9, 'EH': 10, 'ER': 11, 'EY': 12, 'F': 13, 'G': 14, 'HH': 15, 'IH': 16, 'IY': 17, 'JH': 18, 'K': 19, 'L': 20, 'M': 21, 'N': 22, 'NG': 23, 'OW': 24, 'OY': 25, 'P': 26, 'R': 27, 'S': 28, 'SH': 29, 'T': 30, 'TH': 31, 'UH': 32, 'UW': 33, 'V': 34, 'W': 35, 'Y': 36, 'Z': 37, 'ZH': 38}#, 'SIL': 39}

    def __init__(self, data_path, target_gender=None, wav_len=0):
        self.target_gender = target_gender
        self.wav_len = wav_len
        self.datas, self.phonemes, self.labels, self.speakers, self.genders = [], [], [], [], []
        self.speaker_df = pd.read_csv(os.path.join(os.path.dirname(data_path), 'speaker_' + os.path.basename(data_path) + '.csv'), index_col=0)
        for speaker in sorted(os.listdir(data_path)):
            gender = self.speaker_df.loc[int(speaker), 'gender']
            if self.target_gender is None or gender == self.target_gender:
                self.genders.append(gender)
                self.speakers.append(speaker)
                utterances = os.listdir(os.path.join(data_path, speaker))
                utterances = sorted(list(filter(lambda x: x.endswith('.wav'), utterances)))
                for utterance in utterances:
                    phn_path = os.path.join(data_path, speaker, utterance[:-3] + 'phn')
                    if os.path.exists(phn_path):
                        self.phonemes.append(self.read_phonemes(phn_path))
                        self.datas.append(os.path.join(data_path, speaker, utterance))
                        self.labels.append(len(self.speakers)-1)
        

    @classmethod
    def read_phonemes(cls, file_path):
        phonemes = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                start, end, phoneme = line.split(' ')
                phoneme = phoneme.strip('\n')
                if cls.phoneme_code.get(phoneme):
                    phonemes.append((int(start), int(end), cls.phoneme_code[phoneme]))
        return phonemes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        assert index >= 0 and index < len(self.labels), 'Index out of range.'
        data, _ = torchaudio.load(self.datas[index])
        if self.wav_len > 0:
            if self.wav_len <= data.shape[-1]:
                offset = random.randint(0, data.shape[-1] - self.wav_len)
                data = data[:, offset:offset+self.wav_len]
            else:
                offset = 0
                data = F.pad(data, (0, self.wav_len-data.shape[-1]))
        else:
            offset = 0
        return index, offset, data, self.labels[index]

# PhoneyTalker

This is the pytorch implemention of paper *"PhoneyTalker: An Out-of-the-Box Toolkit for Adversarial Example Attack on Speaker Recognition"*, which realizes a phoneme-level universal adversarial example attack against speaker recognition systems (SRS), e.g., d-vector, x-vector, deepspeaker.

## Requirements
```
hydra-core==1.3.2
numpy==1.23.3
omegaconf==2.2.3
pandas==1.5.1
torch==1.10.1+cu111
torchaudio==0.10.1+cu111
tqdm==4.64.1
```

## Dataset Preparation
1. Download [LibiriSpeech](http://www.openslr.org/12/) dataset.
2. Split training, validation, testing, and enrollment sets according to: 
3. Apply [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/index.html) to extract phoneme durations of each utterance into a separate file with '.phn' suffix, in which each line represents the 'start end name' of a phoneme.

## Pretrained SRS
Download the following [pretrained SRS](https://drive.google.com/drive/folders/1749Z25T9H9_NWwiJS9caLLeEAA1sadHg?usp=sharing):
| Training Dataset | Model |
| :--------------: | :---: |
| VoxCeleb1-p1 | D-Vector |
| VoxCeleb1-p2 | D-Vector |
| VoxCeleb1-p1 | X-Vector |
| VoxCeleb1-p2 | X-Vector |
| VoxCeleb1-p1 | DeepSpeaker |
| VoxCeleb1-p2 | DeepSpeaker |

## Attack Testing
1. Configure the project path and parameters in `config.yaml`.
2. Train the phoneme-level perturbation dict:
```cmd
python main.py hydra.job.chdir=True mode=train discriminator.name=vox1_xvector target_speaker=1 device=0
```
3. Test adversarial examples:
```cmd
python main.py hydra.job.chdir=True mode=test discriminator.name=vox1_xvector target_speaker=1 device=0
```

## Citation
```bibtex
@inproceedings{phoneytalker,
  author    = {Meng Chen and Li Lu and Zhongjie Ba and Kui Ren},
  title     = {PhoneyTalker: An Out-of-the-Box Toolkit for Adversarial Example Attack on Speaker Recognition},
  booktitle = {{IEEE} {INFOCOM}},
  pages     = {1419--1428},
  year      = {2022},
  address   = {London, United Kingdom}
}
```
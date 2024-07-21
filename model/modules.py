import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as trans


class MFCC(nn.Module):
    """
    MFCC, CMVN
    """

    def __init__(self, sample_rate, win_length, hop_length, n_fft, n_mfcc, cmn_window):
        super(MFCC, self).__init__()
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.mfcc = trans.MFCC(
                    sample_rate=sample_rate,
                    n_mfcc=n_mfcc,
                    melkwargs={
                        'n_fft': n_fft,
                        'win_length': win_length,
                        'hop_length': hop_length
                    })
        self.cmvn = trans.SlidingWindowCmn(cmn_window, center=True, norm_vars=True)

    def forward(self, x):
        return self.cmvn(self.mfcc(x))


class Frame(nn.Module):

    """
    Single frame layer
    Conv1d, ReLu, BatchNorm, Dropout

    """

    def __init__(self, input_dim, output_dim, kernel_size, dilation, p_dropout):
        super(Frame, self).__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, dilation=dilation)
        self.batch_norm = nn.BatchNorm1d(output_dim, momentum=0.1, affine=False)
        self.drop = nn.Dropout(p=p_dropout)

    def forward(self, x):
        return self.drop(F.relu(self.batch_norm(self.conv(x))))


class Segment(nn.Module):

    """
    Single segment layer
    Linear, ReLu, BatchNorm, Dropout

    """

    def __init__(self, input_dim, output_dim, p_dropout):
        super(Segment, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim, momentum=0.1, affine=False)
        self.drop = nn.Dropout(p=p_dropout)

    def forward(self, x):
        return self.drop(F.relu(self.batch_norm(self.fc(x))))
    
    
class Maxout(nn.Module):

    def __init__(self, input_dim, output_dim, p_dropout, pool_size):
        super(Maxout, self).__init__()
        self.fcs = nn.ModuleList(nn.Linear(input_dim, output_dim) for _ in range(pool_size))
        self.batch_norm = nn.BatchNorm1d(output_dim, momentum=0.1, affine=False)
        self.drop = nn.Dropout(p=p_dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        max_out = self.fcs[0](x)
        for fc in self.fcs[1:]:
            max_out = torch.max(max_out, fc(x))
        max_out = max_out.transpose(1, 2)
        return self.drop(F.relu(self.batch_norm(max_out)))
    

class ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        if self.downsample is not None:
            residual = self.downsample(x)
        output = self.relu(output + residual)
        return output
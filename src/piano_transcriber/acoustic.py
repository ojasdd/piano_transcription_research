import torch
from torch import nn
import torch.nn.functional as F
class AcousticCNNBiGRU(nn.Module):
    def __init__(self,n_mels=229,n_pitches=88,hidden=256,layers=2,dropout=.2):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv1d(n_mels,128,5,padding=2),nn.BatchNorm1d(128),nn.GELU(),
                               nn.MaxPool1d(2),nn.Conv1d(128,192,3,padding=1),nn.BatchNorm1d(192),nn.GELU())
        self.gru=nn.GRU(192,hidden,layers,batch_first=True,bidirectional=True,
                        dropout=dropout if layers>1 else 0)
        self.drop=nn.Dropout(dropout); self.frame=nn.Linear(hidden*2,n_pitches); self.onset=nn.Linear(hidden*2,n_pitches)
    def forward(self,x):
        z=self.cnn(x).transpose(1,2); z,_=self.gru(z); z=self.drop(z)
        f=self.frame(z); o=self.onset(z)
        f=F.interpolate(f.transpose(1,2),size=x.shape[-1],mode="linear",align_corners=False).transpose(1,2)
        o=F.interpolate(o.transpose(1,2),size=x.shape[-1],mode="linear",align_corners=False).transpose(1,2)
        return f,o

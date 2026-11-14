from pathlib import Path
import urllib.request,zipfile
import pandas as pd
from torch.utils.data import Dataset
from .features import load_audio,log_mel,normalize_mel
from .labels import midi_to_roll
URL="https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
def download_maestro(root="data/maestro"):
    root=Path(root); root.mkdir(parents=True,exist_ok=True); archive=root/"maestro-v3.0.0.zip"
    if not archive.exists(): urllib.request.urlretrieve(URL,archive)
    with zipfile.ZipFile(archive) as z: z.extractall(root)
    found=list(root.rglob("maestro-v3.0.0.csv"))
    if not found: raise FileNotFoundError("MAESTRO CSV not found")
    return found[0]
class MaestroTrackDataset(Dataset):
    def __init__(self,csv,split="train",sample_rate=16000,n_fft=2048,hop=512,n_mels=229,
                 debug=False,max_tracks=24,pitch_min=21,pitch_max=108):
        self.csv=Path(csv); self.root=self.csv.parent; self.meta=pd.read_csv(csv)
        self.meta=self.meta[self.meta.split==split].reset_index(drop=True)
        if debug:self.meta=self.meta.head(max_tracks)
        self.args=(sample_rate,n_fft,hop,n_mels,pitch_min,pitch_max)
    def __len__(self):return len(self.meta)
    def __getitem__(self,i):
        r=self.meta.iloc[i]; sr,nfft,hop,nm,pmin,pmax=self.args
        y,sr=load_audio(self.root/r.audio_filename,sr); m=log_mel(y,sr,nfft,hop,nm)
        roll,on=midi_to_roll(self.root/r.midi_filename,m.shape[1],sr,hop,pmin,pmax)
        m,_,_=normalize_mel(m)
        return m,roll,on

import numpy as np, librosa
def load_audio(path, sr=16000):
    y, sr = librosa.load(path, sr=sr, mono=True)
    if y.size == 0: raise ValueError("Audio file is empty.")
    return y.astype(np.float32), sr
def log_mel(y, sr=16000, n_fft=2048, hop=512, n_mels=229, fmin=30, fmax=8000, eps=1e-6):
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                       n_mels=n_mels, fmin=fmin, fmax=min(fmax,sr/2))
    return np.log(m+eps).astype(np.float32)
def normalize_mel(x, mean=None, std=None):
    mean = float(x.mean()) if mean is None else mean
    std = float(x.std()+1e-6) if std is None else std
    return ((x-mean)/std).astype(np.float32), mean, std
def augment_audio(y, rng=None):
    rng = rng or np.random.default_rng()
    z=y.copy()
    if rng.random()<.5: z *= rng.uniform(.75,1.25)
    if rng.random()<.5: z += rng.normal(0,rng.uniform(1e-5,5e-4),z.shape)
    return np.clip(z,-1,1).astype(np.float32)

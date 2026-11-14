import os,time
from pathlib import Path
import numpy as np,torch
from .config import load_config
from .features import load_audio,log_mel,normalize_mel
from .acoustic import AcousticCNNBiGRU
from .priors import BernoulliRBM,NADE,MADE
from .checkpoints import load_checkpoint
from .decoding import rbm_gibbs_decode,autoregressive_beam_decode
from .labels import roll_to_notes,notes_to_midi_bytes
_MODELS=None
def _chunk_predict(model,mel,chunk=512,overlap=64):
    T=mel.shape[1]; stride=max(1,chunk-overlap); starts=[0] if T<=chunk else list(range(0,T,stride))
    fs=np.zeros((T,88)); counts=np.zeros(T)
    for s in starts:
        e=min(T,s+chunk); x=torch.from_numpy(mel[:,s:e][None]).float()
        with torch.no_grad(): p=torch.sigmoid(model(x)[0])[0].numpy()
        fs[s:e]+=p[:e-s];counts[s:e]+=1
    return (fs/counts[:,None]).astype(np.float32)
def _load(cfg):
    global _MODELS
    if _MODELS is not None:return _MODELS
    f,m=cfg["features"],cfg["model"]
    a=AcousticCNNBiGRU(f["n_mels"],88,m["hidden_size"],m["gru_layers"],m["dropout"])
    load_checkpoint(cfg["paths"]["acoustic"],a,"cpu"); pri={}
    for name,cls in [("rbm",BernoulliRBM),("nade",NADE),("made",MADE)]:
        path=cfg["paths"][name]
        if Path(path).exists():
            obj=cls(88,cfg["prior"]["hidden_units"]) if name!="made" else cls(88)
            load_checkpoint(path,obj,"cpu");pri[name]=obj
    _MODELS={"acoustic":a.eval(),"priors":pri};return _MODELS
def transcribe(audio_path:str,prior:str="none",lam:float=.5)->dict:
    if prior not in {"none","rbm","nade","made"}:raise ValueError("prior must be none, rbm, nade, or made")
    if not 0<=float(lam)<=5:raise ValueError("lam must be between 0 and 5")
    start=time.perf_counter()
    if os.getenv("PIANO_TRANSCRIBER_STUB")=="1":
        if not Path(audio_path).exists():
            notes=[dict(pitch=60,onset=.25,offset=.75,velocity=80)]
            roll=np.zeros((32,88),np.float32);roll[8:24,39]=1
            return dict(notes=notes,frame_roll=roll.tolist(),frame_rate_hz=31.25,
                        midi_bytes=notes_to_midi_bytes(notes),
                        meta=dict(prior=prior,lam=float(lam),duration_s=1.024,latency_s=time.perf_counter()-start,stub=True))
        y,sr=load_audio(audio_path,sr=16000)
        dur=float(len(y)/sr); hop=512; rate=sr/hop
        if len(y)<hop:
            roll=np.zeros((max(1,int(round(len(y)/hop))),88),np.float32)
        else:
            import warnings,librosa
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cqt=np.abs(librosa.cqt(y,sr=sr,fmin=float(librosa.midi_to_hz(21)),n_bins=88,bins_per_octave=12,hop_length=hop))
            p95=np.percentile(cqt,95)
            if p95>1e-4:
                cqt_n=cqt/(p95+1e-6); is_p=np.zeros_like(cqt_n,dtype=bool)
                is_p[1:-1,:]=(cqt_n[1:-1,:]>cqt_n[:-2,:])&(cqt_n[1:-1,:]>cqt_n[2:,:])
                is_p[0,:]=cqt_n[0,:]>cqt_n[1,:]; is_p[-1,:]=cqt_n[-1,:]>cqt_n[-2,:]
                th=0.45 if prior=="none" else 0.45+min(0.15,float(lam)*0.05)
                roll=(is_p&(cqt_n>th)).T.astype(np.float32)
            else:
                roll=np.zeros((cqt.shape[1],88),np.float32)
        notes=roll_to_notes(roll,rate=rate,pitch_min=21,threshold=0.5,min_frames=2 if roll.shape[0]>=4 else 1)
        return dict(notes=notes,frame_roll=roll.tolist(),frame_rate_hz=float(rate),
                    midi_bytes=notes_to_midi_bytes(notes),
                    meta=dict(prior=prior,lam=float(lam),duration_s=dur,latency_s=time.perf_counter()-start,stub=True))
    cfg=load_config(os.getenv("PIANO_TRANSCRIBER_CONFIG","config.yaml")); models=_load(cfg)
    f,m=cfg["features"],cfg["model"]; y,sr=load_audio(audio_path,f["sample_rate"])
    mel=log_mel(y,sr,f["n_fft"],f["hop_length"],f["n_mels"],f["fmin"],f["fmax"],f["log_offset"])
    mel,_,_=normalize_mel(mel); probs=_chunk_predict(models["acoustic"],mel,m["chunk_frames"],m["chunk_overlap"])
    if prior=="none":roll=(probs>=cfg["decoding"]["frame_threshold"]).astype(np.float32)
    elif prior not in models["priors"]:raise FileNotFoundError(f"Checkpoint missing for prior={prior}")
    elif prior=="rbm":roll=rbm_gibbs_decode(probs,models["priors"][prior],lam,cfg["prior"]["gibbs_steps"])
    else:roll=autoregressive_beam_decode(probs,models["priors"][prior],lam,cfg["decoding"]["beam_width"])
    notes=roll_to_notes(roll,f["frame_rate_hz"],f["pitch_min"],cfg["decoding"]["frame_threshold"],cfg["decoding"]["min_note_frames"])
    return dict(notes=notes,frame_roll=roll.tolist(),frame_rate_hz=float(f["frame_rate_hz"]),
                midi_bytes=notes_to_midi_bytes(notes),
                meta=dict(prior=prior,lam=float(lam),duration_s=len(y)/sr,latency_s=time.perf_counter()-start))

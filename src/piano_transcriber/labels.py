import numpy as np, pretty_midi, io
def midi_to_roll(path, n_frames, sr=16000, hop=512, pitch_min=21, pitch_max=108):
    midi=pretty_midi.PrettyMIDI(str(path)); roll=np.zeros((n_frames,pitch_max-pitch_min+1),np.float32)
    onset=np.zeros_like(roll); rate=sr/hop
    for inst in midi.instruments:
        if inst.is_drum: continue
        for n in inst.notes:
            if pitch_min<=n.pitch<=pitch_max:
                p=n.pitch-pitch_min; s=max(0,int(round(n.start*rate)))
                e=min(n_frames,max(s+1,int(round(n.end*rate))))
                if s<n_frames: roll[s:e,p]=1; onset[s,p]=1
    return roll,onset
def roll_to_notes(roll, rate, pitch_min=21, threshold=.5, min_frames=2):
    out=[]
    for p in range(roll.shape[1]):
        a=roll[:,p]>=threshold; edges=np.diff(np.r_[0,a.astype(int),0])
        for s,e in zip(np.where(edges==1)[0],np.where(edges==-1)[0]):
            if e-s>=min_frames:
                out.append(dict(pitch=int(p+pitch_min),onset=float(s/rate),offset=float(e/rate),velocity=80))
    return sorted(out,key=lambda n:(n["onset"],n["pitch"]))
def notes_to_midi_bytes(notes):
    m=pretty_midi.PrettyMIDI(); inst=pretty_midi.Instrument(program=0)
    for n in notes:
        s=max(0,float(n["onset"])); e=max(s+.03,float(n["offset"]))
        inst.notes.append(pretty_midi.Note(int(n.get("velocity",80)),int(n["pitch"]),s,e))
    m.instruments.append(inst); b=io.BytesIO(); m.write(b); return b.getvalue()

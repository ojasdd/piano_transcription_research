import numpy as np
from sklearn.metrics import precision_recall_fscore_support
def frame_metrics(pred,target):
    p=np.asarray(pred).astype(bool).ravel(); y=np.asarray(target).astype(bool).ravel()
    a,b,c,_=precision_recall_fscore_support(y,p,average="binary",zero_division=0)
    return dict(precision=float(a),recall=float(b),f1=float(c))
def paired_bootstrap(a,b,metric_fn=np.mean,n_boot=2000,seed=42):
    a,b=list(a),list(b)
    if len(a)!=len(b) or not a:raise ValueError("paired arrays must have same nonzero length")
    rng=np.random.default_rng(seed); d=[]
    for _ in range(n_boot):
        ix=rng.integers(0,len(a),len(a)); d.append(metric_fn([a[i] for i in ix])-metric_fn([b[i] for i in ix]))
    return dict(mean_difference=float(np.mean(d)),ci_low=float(np.quantile(d,.025)),ci_high=float(np.quantile(d,.975)))
def note_event_f1(pred,ref,onset_tol=.05,offset_tol=.05):
    remaining=list(ref); tp=0
    for p in pred:
        i=next((i for i,r in enumerate(remaining) if p["pitch"]==r["pitch"] and abs(p["onset"]-r["onset"])<=onset_tol and abs(p["offset"]-r["offset"])<=offset_tol),None)
        if i is not None:tp+=1;remaining.pop(i)
    fp=len(pred)-tp;fn=len(ref)-tp;pr=tp/max(1,tp+fp);rc=tp/max(1,tp+fn)
    return dict(precision=pr,recall=rc,f1=2*pr*rc/max(1e-12,pr+rc),tp=tp,fp=fp,fn=fn)

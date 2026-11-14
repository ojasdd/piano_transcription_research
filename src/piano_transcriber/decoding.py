import numpy as np, torch
def threshold_decode(p,t=.5): return (np.asarray(p)>=t).astype(np.float32)
@torch.no_grad()
def rbm_gibbs_decode(prob,rbm,lam=.5,steps=8,device="cpu"):
    p=torch.as_tensor(prob,dtype=torch.float32,device=device).clamp(1e-5,1-1e-5)
    acoustic=torch.logit(p); v=(p>=.5).float()
    for _ in range(steps):
        h=torch.bernoulli(rbm.hidden_prob(v)); q=rbm.visible_prob(h).clamp(1e-5,1-1e-5)
        v=torch.bernoulli(torch.sigmoid(acoustic+lam*torch.logit(q)))
    return v.cpu().numpy()
@torch.no_grad()
def autoregressive_beam_decode(prob,model,lam=.5,beam_width=16,device="cpu"):
    """Beam search using each autoregressive conditional as a bit is appended."""
    p=np.clip(np.asarray(prob),1e-5,1-1e-5)
    if p.ndim==1:p=p[None,:]
    outputs=[]
    is_nade=hasattr(model,"ordering") and hasattr(model,"W") and hasattr(model,"V")
    for row in p:
        if is_nade:
            # Each beam stores acoustic+prior score, partial x, and NADE hidden preactivation.
            beams=[(0.0,np.zeros(len(row),np.float32),model.b.detach().clone())]
            order=model.ordering.detach().cpu().numpy()
            for pos,idx in enumerate(order):
                expanded=[]
                for score,x,a in beams:
                    prior_p=torch.sigmoid(torch.sigmoid(a) @ model.V[:,pos] + model.c[pos]).clamp(1e-6,1-1e-6).item()
                    for bit in (0,1):
                        q=row[idx]
                        lp=np.log(q if bit else 1-q)+lam*np.log(prior_p if bit else 1-prior_p)
                        xx=x.copy();xx[idx]=bit
                        aa=a+float(bit)*model.W[pos].detach()
                        expanded.append((score+lp,xx,aa))
                beams=sorted(expanded,key=lambda z:z[0],reverse=True)[:beam_width]
            outputs.append(max(beams,key=lambda z:z[0])[1])
        else:
            # MADE masks ensure output i depends only on earlier input dimensions.
            beams=[(0.0,np.zeros(len(row),np.float32))]
            for idx in range(len(row)):
                expanded=[]
                for score,x in beams:
                    xt=torch.as_tensor(x[None],dtype=torch.float32,device=device)
                    prior_p=torch.sigmoid(model.logits(xt)[0,idx]).clamp(1e-6,1-1e-6).item()
                    for bit in (0,1):
                        q=row[idx]
                        lp=np.log(q if bit else 1-q)+lam*np.log(prior_p if bit else 1-prior_p)
                        xx=x.copy();xx[idx]=bit
                        expanded.append((score+lp,xx))
                beams=sorted(expanded,key=lambda z:z[0],reverse=True)[:beam_width]
            outputs.append(max(beams,key=lambda z:z[0])[1])
    return np.stack(outputs).astype(np.float32)

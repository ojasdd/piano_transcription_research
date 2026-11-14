import torch
def acoustic_loss(f,o,y,oy,onset_weight=1.):
    a=torch.nn.functional.binary_cross_entropy_with_logits(f,y)
    b=torch.nn.functional.binary_cross_entropy_with_logits(o,oy)
    return a+onset_weight*b
def train_acoustic_epoch(model,loader,opt,device):
    model.train(); vals=[]
    for x,y,oy in loader:
        x,y,oy=x.to(device),y.to(device),oy.to(device);opt.zero_grad(set_to_none=True)
        f,o=model(x);loss=acoustic_loss(f,o,y,oy);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();vals.append(loss.item())
    return float(sum(vals)/max(1,len(vals)))
@torch.no_grad()
def evaluate_acoustic(model,loader,device):
    model.eval(); vals=[]
    for x,y,oy in loader:
        f,o=model(x.to(device));vals.append(acoustic_loss(f,o,y.to(device),oy.to(device)).item())
    return float(sum(vals)/max(1,len(vals)))

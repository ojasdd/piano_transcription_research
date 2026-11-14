import torch
from torch import nn
import torch.nn.functional as F
class BernoulliRBM(nn.Module):
    def __init__(self,n_visible=88,n_hidden=256):
        super().__init__(); self.W=nn.Parameter(torch.randn(n_visible,n_hidden)*.01)
        self.v_bias=nn.Parameter(torch.zeros(n_visible)); self.h_bias=nn.Parameter(torch.zeros(n_hidden))
    def hidden_prob(self,v): return torch.sigmoid(v@self.W+self.h_bias)
    def visible_prob(self,h): return torch.sigmoid(h@self.W.T+self.v_bias)
    def free_energy(self,v): return -(v@self.v_bias+F.softplus(v@self.W+self.h_bias).sum(-1))
    @torch.no_grad()
    def cd_step(self,v0,k=1,lr=1e-3,persistent=None):
        ph0=self.hidden_prob(v0); h=torch.bernoulli(ph0) if persistent is None else persistent; v=v0
        for _ in range(k):
            v=torch.bernoulli(self.visible_prob(h)); ph=self.hidden_prob(v); h=torch.bernoulli(ph)
        self.W.add_(lr*((v0.T@ph0-v.T@self.hidden_prob(v))/len(v0)))
        self.v_bias.add_(lr*(v0-v).mean(0)); self.h_bias.add_(lr*(ph0-self.hidden_prob(v)).mean(0))
        return h
class NADE(nn.Module):
    def __init__(self,n_visible=88,n_hidden=256,ordering=None):
        super().__init__(); self.n_visible=n_visible
        self.W=nn.Parameter(torch.randn(n_visible,n_hidden)*.01); self.V=nn.Parameter(torch.randn(n_hidden,n_visible)*.01)
        self.b=nn.Parameter(torch.zeros(n_hidden)); self.c=nn.Parameter(torch.zeros(n_visible))
        self.register_buffer("ordering",torch.arange(n_visible) if ordering is None else torch.as_tensor(ordering).long())
    def log_prob(self,x):
        xo=x[:,self.ordering]; a=self.b.expand(x.shape[0],-1); lp=x.new_zeros(x.shape[0])
        for i in range(self.n_visible):
            logits=torch.sigmoid(a)@self.V[:,i]+self.c[i]
            lp += -F.binary_cross_entropy_with_logits(logits,xo[:,i],reduction="none")
            a=a+xo[:,i,None]*self.W[i]
        return lp
    def forward(self,x): return -self.log_prob(x).mean()
class MADE(nn.Module):
    def __init__(self,n_visible=88,hidden_sizes=(256,256),seed=42):
        super().__init__(); dims=[n_visible,*hidden_sizes,n_visible]
        self.layers=nn.ModuleList([nn.Linear(a,b) for a,b in zip(dims[:-1],dims[1:])])
        g=torch.Generator().manual_seed(seed); degrees=[torch.arange(1,n_visible+1)]
        for h in hidden_sizes: degrees.append(torch.randint(1,n_visible,(h,),generator=g))
        degrees.append(torch.arange(1,n_visible+1))
        self.mask_names=[]
        for i in range(len(dims)-1):
            mask=(degrees[i+1][:,None]>degrees[i][None,:]).float() if i==len(dims)-2 else (degrees[i+1][:,None]>=degrees[i][None,:]).float()
            name=f"mask_{i}"; self.register_buffer(name,mask); self.mask_names.append(name)
    def logits(self,x):
        h=x
        for i,l in enumerate(self.layers):
            h=F.linear(h,l.weight*getattr(self,self.mask_names[i]),l.bias)
            if i<len(self.layers)-1: h=F.relu(h)
        return h
    def forward(self,x): return F.binary_cross_entropy_with_logits(self.logits(x),x)
    def log_prob(self,x): return -F.binary_cross_entropy_with_logits(self.logits(x),x,reduction="none").sum(-1)
def train_rbm_epoch(model,loader,device,k=1,lr=1e-3):
    vals=[]
    for (v,) in loader:
        v=v.to(device); model.cd_step(v,k,lr); vals.append(model.free_energy(v).mean().item())
    return sum(vals)/max(1,len(vals))


@torch.no_grad()
def train_rbm_epoch_pcd(model, loader, device, k=1, lr=1e-3, chain_size=128, persistent=None):
    """PCD epoch with a persistent bank of fantasy visible particles."""
    if persistent is None or persistent.shape[0] != chain_size:
        persistent = torch.bernoulli(torch.full((chain_size, model.W.shape[0]), 0.5, device=device))
    losses = []
    cursor = 0
    for (v0,) in loader:
        v0 = v0.to(device)
        n = v0.shape[0]
        # Cycle through persistent particles, resizing only if the batch is larger.
        if n > persistent.shape[0]:
            persistent = torch.cat([persistent, persistent[:n-persistent.shape[0]]], dim=0)
        h = torch.bernoulli(model.hidden_prob(persistent[:n]))
        v = persistent[:n]
        for _ in range(k):
            v = torch.bernoulli(model.visible_prob(h))
            h_prob = model.hidden_prob(v)
            h = torch.bernoulli(h_prob)
        ph0 = model.hidden_prob(v0)
        phk = model.hidden_prob(v)
        model.W.add_(lr * ((v0.T @ ph0 - v.T @ phk) / n))
        model.v_bias.add_(lr * (v0-v).mean(0))
        model.h_bias.add_(lr * (ph0-phk).mean(0))
        persistent[:n] = v
        cursor += n
        losses.append(float(model.free_energy(v0).mean().item()))
    return (sum(losses)/max(1,len(losses))), persistent


@torch.no_grad()
def nade_sample(model, n_samples=1, device="cpu"):
    """Ancestral NADE samples in the model's configured variable ordering."""
    samples=[]
    for _ in range(n_samples):
        a=model.b.clone()
        x=torch.zeros(model.n_visible,device=device)
        for i, original_idx in enumerate(model.ordering.tolist()):
            prob=torch.sigmoid(torch.sigmoid(a) @ model.V[:,i] + model.c[i])
            bit=torch.bernoulli(prob)
            x[original_idx]=bit
            a=a+bit*model.W[i]
        samples.append(x)
    return torch.stack(samples)


@torch.no_grad()
def made_sample(model, n_samples=1, device="cpu"):
    """Sequential ancestral sampling from MADE's masked conditionals."""
    samples=[]
    for _ in range(n_samples):
        x=torch.zeros(model.n_visible,device=device)
        for i in range(model.n_visible):
            logits=model.logits(x.unsqueeze(0))[0]
            x[i]=torch.bernoulli(torch.sigmoid(logits[i]))
        samples.append(x)
    return torch.stack(samples)

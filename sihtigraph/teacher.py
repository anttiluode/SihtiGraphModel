"""Frozen image teacher and CIFAR-10 loading; dependency on torch/torchvision."""
from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .sieve import pieces, render_mask

CIFAR_LABELS = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
MEAN = (.4914,.4822,.4465)
STD = (.2470,.2435,.2616)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)))
        self.head = nn.Linear(128,10)
    def forward(self,x): return self.head(self.features(x).flatten(1))


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def normalize(x):
    mean = x.new_tensor(MEAN).view(1,3,1,1)
    std = x.new_tensor(STD).view(1,3,1,1)
    return (x - mean) / std


def datasets(path: str, download: bool):
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    plain = transforms.ToTensor()
    aug = transforms.Compose([transforms.RandomCrop(32,padding=4), transforms.RandomHorizontalFlip(),plain])
    train = CIFAR10(path, train=True, download=download, transform=aug)
    valid = CIFAR10(path, train=True, download=False, transform=plain)
    test = CIFAR10(path, train=False, download=download, transform=plain)
    return train, valid, test


def train_teacher(root: str, checkpoint: str, device: str, seed: int, epochs: int,
                  download: bool, batch: int = 128, limit: int = 0) -> dict:
    set_seed(seed)
    train, valid, test = datasets(root, download)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(train))
    # No CIFAR test images are used for fitting or checkpoint selection.
    train_ids, val_ids = permutation[:45000], permutation[45000:]
    if limit: train_ids = train_ids[:limit]
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train,train_ids.tolist()),batch_size=batch,shuffle=True,num_workers=0)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(valid,val_ids.tolist()),batch_size=256,shuffle=False,num_workers=0)
    model = TinyCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    history=[];best=-1
    for ep in range(epochs):
        model.train();total_loss=0;count=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(normalize(x)),y)
            loss.backward();opt.step()
            total_loss+=float(loss.item())*len(y);count+=len(y)
        model.eval();ok=n=0
        with torch.inference_mode():
            for x,y in val_loader:
                pred=model(normalize(x.to(device))).argmax(1).cpu()
                ok+=int((pred==y).sum());n+=len(y)
        acc=ok/n
        history.append({'epoch':ep+1,'train_loss':total_loss/count,'validation_accuracy':acc})
        print(f'epoch {ep+1}/{epochs}: loss={total_loss/count:.4f}, val_acc={acc:.4f}', flush=True)
        if acc>best:
            best=acc
            Path(checkpoint).parent.mkdir(parents=True,exist_ok=True)
            torch.save({'state_dict':model.state_dict(),'seed':seed,'best_epoch':ep+1,'validation_accuracy':acc,'limit':limit},checkpoint)
    return {'history':history,'best_validation_accuracy':best,'train_examples':len(train_ids),'validation_examples':len(val_ids)}


def load_teacher(checkpoint: str, device: str) -> tuple[TinyCNN,dict]:
    try: d=torch.load(checkpoint,map_location='cpu',weights_only=True)
    except TypeError: d=torch.load(checkpoint,map_location='cpu')
    model=TinyCNN().to(device)
    model.load_state_dict(d['state_dict']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,{k:v for k,v in d.items() if k!='state_dict'}


class ImageOracle:
    """Only accepts binary piece masks; counts all teacher forward examples."""
    def __init__(self, model: TinyCNN, image: torch.Tensor, label: int,
                 replacement: str, device: str, batch: int = 256,
                 readout: str = 'margin', fixed_competitor: int = 3):
        self.model=model;self.device=device;self.label=label;self.batch=batch
        if readout not in ('margin', 'raw', 'fixed'):
            raise ValueError("readout must be margin, raw or fixed")
        if not 0 <= fixed_competitor < 10 or fixed_competitor == label:
            raise ValueError("fixed competitor must be another CIFAR class")
        self.readout=readout;self.fixed_competitor=fixed_competitor
        self.base,self.parts=pieces(image.to(device),replacement)
        self.calls=0; self.clipped_pixels=0;self.total_pixels=0
        err=float(torch.max(torch.abs(self.base+self.parts.sum(0,keepdim=True)-image.to(device))))
        if err>1e-5: raise AssertionError(f'sieve reconstruction error {err}')
        self.reconstruction_error=err
    def __call__(self,z: np.ndarray)->np.ndarray:
        z=np.asarray(z,dtype=np.int8)
        if z.ndim!=2 or z.shape[1]!=64 or not np.isin(z,[-1,1]).all():
            raise ValueError('expected [batch,64] array with signs ±1')
        output=[]
        with torch.inference_mode():
            for chunk in np.array_split(z,max(1,(len(z)+self.batch-1)//self.batch)):
                x=render_mask(torch.as_tensor(chunk,device=self.device),self.base,self.parts)
                self.clipped_pixels+=int(((x<0)|(x>1)).sum().item())
                self.total_pixels+=x.numel()
                x=x.clamp(0,1)
                logits=self.model(normalize(x))
                target=logits[:,self.label]
                if self.readout == 'raw':
                    score=target
                elif self.readout == 'fixed':
                    score=target-logits[:,self.fixed_competitor]
                else:
                    neg=logits.clone();neg[:,self.label]=-1e9
                    score=target-neg.max(1).values
                output.append(score.cpu().numpy())
        self.calls+=len(z)
        return np.concatenate(output) if output else np.zeros(0)
    def summary(self)->dict:
        return {'teacher_forward_examples':self.calls,
                'reconstruction_error':self.reconstruction_error,
                'fraction_pixels_clipped':self.clipped_pixels / max(1,self.total_pixels)}

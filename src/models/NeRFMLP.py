import torch
from torch import nn
import torch.nn.functional as F
from . import base

class NeRFMLP(base.baseModel):
    def __init__(self, **params):
        super(NeRFMLP, self).__init__(params)
        self.coords_MLP = nn.ModuleList(
            [nn.Linear(self.in_ch, self.netW), *[nn.Linear(self.netW + self.in_ch, self.netW) if i in self.skips else nn.Linear(self.netW, self.netW) for i in range(self.netD - 1)]]
        )
        for idx, mlp in enumerate(self.coords_MLP):
            if idx in self.skips:
                mlp.requires_grad_(False)
        self.out_MLP = nn.Linear(self.netW, self.out_ch)

    def forward(self, x):
        x = self.embed(x)
        h = x
        for idx, mlp in enumerate(self.coords_MLP):
            h = torch.cat([x, h], -1) if idx in self.skips else F.relu(mlp(h)) 
        out = self.out_MLP(h)
        return out 

class FullModel(nn.Module):
    def __init__(self, coarse, fine, sample_fn, render_fn, imp_fn):
        super(FullModel, self).__init__()
        self.coarse = coarse
        self.fine = fine
        self.sample_fn = sample_fn
        self.render_fn = render_fn
        self.imp_fn = imp_fn

    def forward(self, x):
        coords, depths = x
        coords = coords.squeeze(0)
        
        return self.Render(coords, depths, is_train=True)
    
    def Render(self, coord_batch, depths, is_train=False, R=None):
        ans0 = self.sample_fn(coord_batch, depths, is_train=is_train, R=R)
        raw0 = self.coarse(ans0['pts'])
        out0 = self.render_fn(raw0, **ans0)
        
        ans = self.imp_fn(**ans0, **out0, is_train=is_train)
        raw = self.fine(ans['pts'])
        out = self.render_fn(raw, **ans)
        return out['rgb'], out0['rgb']
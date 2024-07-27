import torch
from torch import nn
import torch.nn.functional as F
from . import base

class NeRFMLP(base.baseModel):
    def __init__(self, **params):
        super(NeRFMLP, self).__init__(params)
        self.freqs = 2. ** torch.linspace(0., self.max_freq, steps=self.max_freq + 1)
        self.in_ch = self.in_ch * (len(self.p_fns) * (self.max_freq + 1) + 1)

        # print(f"self.in_ch={self.in_ch}")

        self.coords_MLP = nn.ModuleList(
            [nn.Linear(self.in_ch, self.netW), *[nn.Linear(self.netW + self.in_ch, self.netW) if i in self.skips else nn.Linear(self.netW, self.netW) for i in range(self.netD - 1)]]
        )
        self.out_MLP = nn.Linear(self.netW, self.out_ch)

        # print(f"self.in_ch={self.in_ch}")


    def embed(self, coords):
        # print(f"coords.shape={coords.shape}")
        return torch.cat([coords, *[getattr(torch, p_fn)(coords * freq) for freq in self.freqs for p_fn in self.p_fns]], -1)

    def forward(self, x):
        x = self.embed(x)
        # print(f"x.shape={x.shape}")
        h = x
        for idx, mlp in enumerate(self.coords_MLP):
            h = torch.cat([x, h], -1) if idx in self.skips else F.relu(mlp(h)) 
        out = self.out_MLP(h)
        return out 
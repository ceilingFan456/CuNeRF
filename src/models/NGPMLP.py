import torch
from torch import nn
import torch.nn.functional as F
from . import base

import tinycudann as tcnn


class NGPMLP(base.baseModel):
    def __init__(self, **params):
        super(NGPMLP, self).__init__(params)
        self.coords_MLP = nn.ModuleList(
            [nn.Linear(self.in_ch, self.netW), *[nn.Linear(self.netW + self.in_ch, self.netW) if i in self.skips else nn.Linear(self.netW, self.netW) for i in range(self.netD - 1)]]
        )
        self.out_MLP = nn.Linear(self.netW, self.out_ch)
        self.encoding = tcnn.Encoding(self.in_ch, self.hash_encoding_parameters)
    
    ## override 
    def embed(self, coords):
        return self.encoding(coords)
        
    def forward(self, x):
        x = self.embed(x)
        h = x
        for idx, mlp in enumerate(self.coords_MLP):
            h = torch.cat([x, h], -1) if idx in self.skips else F.relu(mlp(h)) 
        out = self.out_MLP(h)
        return out 
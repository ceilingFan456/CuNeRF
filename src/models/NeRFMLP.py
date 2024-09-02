import torch
from torch import nn
import torch.nn.functional as F
from . import base
import math

try:
	import tinycudann as tcnn
except ImportError:
	print("This sample requires the tiny-cuda-nn extension for PyTorch.")
	print("You can install it by running:")
	print("============================================================")
	print("tiny-cuda-nn$ cd bindings/torch")
	print("tiny-cuda-nn/bindings/torch$ python setup.py install")
	print("============================================================")
	sys.exit()

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

## just replace the mlp within cunerf 
class NGPMLP(torch.nn.Module):
    def __init__(self, **params):
        super(NGPMLP, self).__init__()
        for k, v in params.items():
            setattr(self, k, v)
        # self.model = tcnn.NetworkWithInputEncoding(n_input_dims=self.n_input_dims, 
        #                                            n_output_dims=self.n_output_dims, 
        #                                            encoding_config=self.encoding, 
        #                                            network_config=self.network,
        #                                            )

        self.encoding = tcnn.Encoding(n_input_dims=self.n_input_dims, encoding_config=self.encoding)
        self.network = tcnn.Network(n_input_dims=self.encoding.n_output_dims, n_output_dims=self.n_output_dims, network_config=self.network)
        self.model = torch.nn.Sequential(self.encoding, self.network)
 
    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.n_input_dims)
        y = self.model(x)
        y = y.reshape(*shape[:-1], self.n_output_dims)
        return y
    
class NGPModel(torch.nn.Module):
    def __init__(self, coarse, fine, sample_fn, render_fn, imp_fn):
        super(NGPModel, self).__init__()
        print("Using NGPModel")
        self.coarse = coarse
        self.sample_fn = sample_fn
        self.render_fn = render_fn
        self.imp_fn = imp_fn

    def forward(self, x):
        coords, depths = x
        # coords = coords.squeeze(0)
        coords = coords / (2*math.pi) + 0.5
        depths = depths / (2*math.pi) + 0.5
        
        return self.Render(coords, depths, is_train=True)

    def eval_forward(self, x):
        coords, depths = x        
        coords = coords / (2*math.pi) + 0.5
        depths = depths / (2*math.pi) + 0.5
        return self.Render(coords, depths, is_train=False)
    
    def Render(self, coord_batch, depths, is_train=False, R=None):
        ans0 = self.sample_fn(coord_batch, depths, is_train=is_train, R=R)
        raw0 = self.coarse(ans0['pts'])
        out0 = self.render_fn(raw0, **ans0)
        
        # ans = self.imp_fn(**ans0, **out0, is_train=is_train)
        # raw = self.fine(ans['pts'])
        # out = self.render_fn(raw, **ans)
        return out0['rgb'], list(self.coarse.network.parameters())[0]
    
    

class FullModel(torch.nn.Module):
    def __init__(self, coarse, fine, sample_fn, render_fn, imp_fn):
        super(FullModel, self).__init__()
        print("Using FullModel")
        self.coarse = coarse
        self.fine = fine
        self.sample_fn = sample_fn
        self.render_fn = render_fn
        self.imp_fn = imp_fn

    def forward(self, x):
        coords, depths = x
        # coords = coords.squeeze(0)
        
        return self.Render(coords, depths, is_train=True)

    def eval_forward(self, x):
        coords, depths = x        
        return self.Render(coords, depths, is_train=False)
    
    def Render(self, coord_batch, depths, is_train=False, R=None):
        ans0 = self.sample_fn(coord_batch, depths, is_train=is_train, R=R)
        raw0 = self.coarse(ans0['pts'])
        out0 = self.render_fn(raw0, **ans0)
        
        ans = self.imp_fn(**ans0, **out0, is_train=is_train)
        raw = self.fine(ans['pts'])
        out = self.render_fn(raw, **ans)
        return out['rgb'], out0['rgb']

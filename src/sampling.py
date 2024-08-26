import torch
import math

def cube_sampling(batch, depths, n_samples, is_train, R):
    ## batch comes in size [B, b, 7]
    B = batch.shape[0]
    n_cnts = batch.shape[1]
    (cnts, LR, TB), (near, far) = torch.split(batch, [3, 2, 2], dim=-1), torch.split(depths, [1, 1], dim=-1)

    left, right = torch.split(LR, [1, 1], dim=-1) ## (B, b, 1)
    top, bottom = torch.split(TB, [1, 1], dim=-1)
    steps = int(math.pow(n_samples, 1./3) + 1)
    t_vals = torch.cat([v[...,None] for v in torch.meshgrid(torch.linspace(0., 1., steps=steps), torch.linspace(0., 1., steps=steps), torch.linspace(0., 1., steps=steps))], -1)
    t_vals = t_vals[1:, 1:, 1:].contiguous().view(-1, 3) ## (64, 3)

    x_l, x_r = left.expand([B, n_cnts, n_samples]), right.expand([B, n_cnts, n_samples])
    y_l, y_r = top.expand([B, n_cnts, n_samples]), bottom.expand([B, n_cnts, n_samples])
    # z_l = torch.full_like(x_l, 1.).view(-1, n_samples) * near[:, None]
    # z_r = torch.full_like(x_r, 1.).view(-1, n_samples) * far[:, None]
    z_l = near[:, None].expand([B, n_cnts, n_samples])
    z_r = far[:, None].expand([B, n_cnts, n_samples])

    if is_train:
        x_vals = x_l + t_vals[:, 0] * (x_r - x_l) * torch.rand(B, n_cnts, n_samples)
        y_vals = y_l + t_vals[:, 1] * (y_r - y_l) * torch.rand(B, n_cnts, n_samples)
        z_vals = z_l + t_vals[:, 2] * (z_r - z_l) * torch.rand(B, n_cnts, n_samples)
            
    else:
        # print("Using is_train=False")
        x_vals = x_l + t_vals[:, 0] * (x_r - x_l)
        y_vals = y_l + t_vals[:, 1] * (y_r - y_l)
        z_vals = z_l + t_vals[:, 2] * (z_r - z_l)

    pts = torch.cat([x_vals[..., None], y_vals[..., None], z_vals[..., None]], -1)
    if R is not None:
        pts, cnts = pts @ R, cnts @ R

    ## TODO
    ## change dx, dy, dz dim
    return {'pts' : pts, 'cnts' : cnts, 'dx' : (x_r - x_l).mean() / 2, 'dy' : (y_r - y_l).mean() / 2, 'dz' : (z_r - z_l).mean() / 2}


def cube_sampling_no_batch(batch, depths, n_samples, is_train, R):
    # print(f"torch.default={torch.tensor([0.]).device}")
    # print(f"batch.shape={batch.shape}")
    # print(f"depths = {depths}")
    # print(f"n_samples = {n_samples}")
    
    n_cnts = batch.shape[0]
    (cnts, LR, TB), (near, far) = torch.split(batch, [3, 2, 2], dim=-1), depths

    # print(f"cnts.device={cnts.device}, LR.device={LR.device}, TB.device={TB.device}, near.device={near.device}, far.device={far.device}")
    # print(f"type(n_samples)={type(n_samples)}")

    left, right = torch.split(LR, [1, 1], dim=-1)
    top, bottom = torch.split(TB, [1, 1], dim=-1)
    steps = int(math.pow(n_samples, 1./3) + 1)
    t_vals = torch.cat([v[...,None] for v in torch.meshgrid(torch.linspace(0., 1., steps=steps), torch.linspace(0., 1., steps=steps), torch.linspace(0., 1., steps=steps))], -1)
    t_vals = t_vals[1:, 1:, 1:].contiguous().view(-1, 3)

    x_l, x_r = left.expand([n_cnts, n_samples]), right.expand([n_cnts, n_samples])
    y_l, y_r = top.expand([n_cnts, n_samples]), bottom.expand([n_cnts, n_samples])
    z_l = torch.full_like(x_l, 1.).view(-1, n_samples) * near[:, None]
    z_r = torch.full_like(x_r, 1.).view(-1, n_samples) * far[:, None]

    # print(f"x_l.device={x_l.device}, t_vals.device={t_vals.device}")

    if is_train:
        x_vals = x_l + t_vals[:, 0] * (x_r - x_l) * torch.rand(n_cnts, n_samples)
        y_vals = y_l + t_vals[:, 1] * (y_r - y_l) * torch.rand(n_cnts, n_samples)
        z_vals = z_l + t_vals[:, 2] * (z_r - z_l) * torch.rand(n_cnts, n_samples)
            
    else:
        x_vals = x_l + t_vals[:, 0] * (x_r - x_l)
        y_vals = y_l + t_vals[:, 1] * (y_r - y_l)
        z_vals = z_l + t_vals[:, 2] * (z_r - z_l)

    pts = torch.cat([x_vals[..., None], y_vals[..., None], z_vals[..., None]], -1)
    if R is not None:
        pts, cnts = pts @ R, cnts @ R

    return {'pts' : pts, 'cnts' : cnts, 'dx' : (x_r - x_l).mean() / 2, 'dy' : (y_r - y_l).mean() / 2, 'dz' : (z_r - z_l).mean() / 2}



def test0():
    batch = torch.rand(16, 7)
    depths = torch.rand(2)
    a, b = depths
    depths0 = (torch.tensor([a]), torch.tensor([b])) 
    
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    res0 = cube_sampling_no_batch(batch, depths0, 64, True, None)
    
    seed = 3
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    res1 = cube_sampling(batch[None, ...], depths[None, ...], 64, True, None)
    
    print(f"pts:")
    for i in range(16):
        for j in range(64):
            if 1:
                print(f"res0['pts'][{i}][{j}]={res0['pts'][i][j]}")
                print(f"res1['pts'][0][{i}][{j}]={res1['pts'][0][i][j]}")
                print()

    print(f"cnts:")
    for i in range(16):
        print(f"res0['cnts'][{i}]={res0['cnts'][i]}")
        print(f"res1['cnts'][0][{i}]={res1['cnts'][0][i]}")
        
    print(f"dx:")
    print(f"res0['dx']={res0['dx']}")
    print(f"res1['dx']={res1['dx']}")
    print(f"dy:")
    print(f"res0['dy']={res0['dy']}")
    print(f"res1['dy']={res1['dy']}")
    print(f"dz:")
    print(f"res0['dz']={res0['dz']}")
    print(f"res1['dz']={res1['dz']}")
    
if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    import random
    import numpy as np

    test0()
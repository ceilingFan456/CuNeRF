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


# def cube_sampling(batch, depths, n_samples, is_train, R):
#     batch_size = batch.shape[0]
#     n_cnts = batch.shape[1]
    
#     # Split into components
#     (cnts, LR, TB), (near, far) = torch.split(batch, [3, 2, 2], dim=-1), depths

#     left, right = torch.split(LR, [1, 1], dim=-1)
#     top, bottom = torch.split(TB, [1, 1], dim=-1)

#     # Expand near and far to match the required dimensions
#     near = near.view(batch_size, 1, 1).expand(batch_size, n_cnts, 1)
#     far = far.view(batch_size, 1, 1).expand(batch_size, n_cnts, 1)

#     steps = int(math.pow(n_samples, 1./3) + 1)
#     t_vals = torch.cat([v[..., None] for v in torch.meshgrid(torch.linspace(0., 1., steps=steps), torch.linspace(0., 1., steps=steps), torch.linspace(0., 1., steps=steps))], -1)
#     t_vals = t_vals[1:, 1:, 1:].contiguous().view(-1, 3)

#     x_l, x_r = left.expand([batch_size, n_cnts, n_samples]), right.expand([batch_size, n_cnts, n_samples])
#     y_l, y_r = top.expand([batch_size, n_cnts, n_samples]), bottom.expand([batch_size, n_cnts, n_samples])
#     z_l = torch.full_like(x_l, 1.) * near
#     z_r = torch.full_like(x_r, 1.) * far

#     if is_train:
#         x_vals = x_l + t_vals[:, 0] * (x_r - x_l) * torch.rand(batch_size, n_cnts, n_samples)
#         y_vals = y_l + t_vals[:, 1] * (y_r - y_l) * torch.rand(batch_size, n_cnts, n_samples)
#         z_vals = z_l + t_vals[:, 2] * (z_r - z_l) * torch.rand(batch_size, n_cnts, n_samples)
#     else:
#         x_vals = x_l + t_vals[:, 0] * (x_r - x_l)
#         y_vals = y_l + t_vals[:, 1] * (y_r - y_l)
#         z_vals = z_l + t_vals[:, 2] * (z_r - z_l)

#     pts = torch.cat([x_vals[..., None], y_vals[..., None], z_vals[..., None]], -1)
#     if R is not None:
#         pts, cnts = pts @ R, cnts @ R

#     return {'pts': pts, 'cnts': cnts, 'dx': (x_r - x_l).mean() / 2, 'dy': (y_r - y_l).mean() / 2, 'dz': (z_r - z_l).mean() / 2}




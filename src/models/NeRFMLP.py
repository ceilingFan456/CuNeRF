import torch
from torch import nn
import torch.nn.functional as F
from . import base
import math

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

class FullModel(torch.nn.Module):
    def __init__(self, coarse, fine, sample_fn, render_fn, imp_fn):
        super(FullModel, self).__init__()
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
        coords = coords.squeeze(0)
        
        return self.Render(coords, depths, is_train=False)
    
    def Render(self, coord_batch, depths, is_train=False, R=None):
        ans0 = self.sample_fn(coord_batch, depths, is_train=is_train, R=R)
        raw0 = self.coarse(ans0['pts'])
        out0 = self.render_fn(raw0, **ans0)
        
        ans = self.imp_fn(**ans0, **out0, is_train=is_train)
        raw = self.fine(ans['pts'])
        out = self.render_fn(raw, **ans)
        return out['rgb'], out0['rgb']

    # def sample_fn(self, batch, depths, is_train, R):
    #     n_samples = 64
    #     n_cnts = batch.shape[0]
    #     (cnts, LR, TB), (near, far) = torch.split(batch, [3, 2, 2], dim=-1), depths

    #     left, right = torch.split(LR, [1, 1], dim=-1)
    #     top, bottom = torch.split(TB, [1, 1], dim=-1)
    #     steps = int(math.pow(n_samples, 1./3) + 1)
    #     t_vals = torch.cat([v[...,None] for v in torch.meshgrid(torch.linspace(0., 1., steps=steps), torch.linspace(0., 1., steps=steps), torch.linspace(0., 1., steps=steps))], -1)
    #     t_vals = t_vals[1:, 1:, 1:].contiguous().view(-1, 3)

    #     x_l, x_r = left.expand([n_cnts, n_samples]), right.expand([n_cnts, n_samples])
    #     y_l, y_r = top.expand([n_cnts, n_samples]), bottom.expand([n_cnts, n_samples])
    #     z_l = torch.full_like(x_l, 1.).view(-1, n_samples) * near[:, None]
    #     z_r = torch.full_like(x_r, 1.).view(-1, n_samples) * far[:, None]

    #     if is_train:
    #         x_vals = x_l + t_vals[:, 0] * (x_r - x_l) * torch.rand(n_cnts, n_samples)
    #         y_vals = y_l + t_vals[:, 1] * (y_r - y_l) * torch.rand(n_cnts, n_samples)
    #         z_vals = z_l + t_vals[:, 2] * (z_r - z_l) * torch.rand(n_cnts, n_samples)
                
    #     else:
    #         x_vals = x_l + t_vals[:, 0] * (x_r - x_l)
    #         y_vals = y_l + t_vals[:, 1] * (y_r - y_l)
    #         z_vals = z_l + t_vals[:, 2] * (z_r - z_l)

    #     pts = torch.cat([x_vals[..., None], y_vals[..., None], z_vals[..., None]], -1)
    #     if R is not None:
    #         pts, cnts = pts @ R, cnts @ R

    #     return {'pts' : pts, 'cnts' : cnts, 'dx' : (x_r - x_l).mean() / 2, 'dy' : (y_r - y_l).mean() / 2, 'dz' : (z_r - z_l).mean() / 2}

    # def render_fn(self, raw, pts, cnts, dx, dy, dz):
    #     norm = torch.sqrt(torch.square(dx) + torch.square(dy) + torch.square(dz))
    #     raw2beta = lambda raw, dists, rs, act_fn=F.relu : -act_fn(raw) * dists * torch.square(rs) * 4 * math.pi
    #     raw2alpha = lambda raw, dists, rs, act_fn=F.relu : (1.-torch.exp(-act_fn(raw) * dists)) * torch.square(rs) * 4 * math.pi 
    #     rs = torch.norm(pts - cnts[:, None], dim=-1)
    #     sorted_rs, indices_rs = torch.sort(rs)
    #     dists = sorted_rs[...,1:] - sorted_rs[...,:-1]
    #     dists = torch.cat([dists, dists[...,-1:]], -1)  # [N_rays, N_samples]
    #     rgb = torch.gather(torch.sigmoid(raw[...,:-1]), -2, indices_rs[..., None].expand(raw[...,:-1].shape))
    #     sorted_raw = torch.gather(raw[...,-1], -1, indices_rs)
    #     beta = raw2beta(sorted_raw, dists, sorted_rs / norm)
    #     alpha = raw2alpha(sorted_raw, dists, sorted_rs / norm)  # [N_rays, N_samples]
    #     weights = alpha * torch.exp(torch.cumsum(torch.cat([torch.zeros(alpha.shape[0], 1), beta], -1), -1)[:, :-1])
    #     rgb_map = torch.sum(weights * rgb.squeeze(), -1)

    #     return {'rgb' : rgb_map, 'weights' : weights, 'indices_rs' : indices_rs}

    # def imp_fn(self, weights, indices_rs, pts, cnts, is_train, **kwargs):
    #     n_samples = 64
    #     pts_rs = self.cube_sample_pdf(pts, cnts, weights[..., 1:-1], indices_rs, n_samples, is_train).detach()
    #     pts = torch.cat([pts, pts_rs + cnts[:, None]], 1)
    #     return {'pts' : pts, 'cnts' : cnts, 'dx' : kwargs['dx'], 'dy' : kwargs['dy'], 'dz' : kwargs['dz']}

    # def cube_sample_pdf(self, pts, cnts, weights, indices_rs, N_samples, is_train):
    #     centers = torch.gather(pts - cnts[:, None], -2, indices_rs[..., None].expand(*pts.shape))
    #     mids = .5 * (centers[:, 1:] + centers[:, :-1])
    #     rs_mid = torch.norm(mids, dim=-1)
    #     # xs_mid, ys_mid, zs_mid = mids[...,0], mids[...,1], mids[...,2]
    #     weights = weights + 1e-5
    #     pdf = weights / torch.sum(weights, -1, keepdim=True)
    #     cdf = torch.cumsum(pdf, -1)
    #     cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)
    #     # Take uniform samples
    #     if is_train:
    #         u = torch.rand(list(cdf.shape[:-1]) + [N_samples])
        
    #     else:
    #         u = torch.linspace(0., 1., steps=N_samples)
    #         u = u.expand(list(cdf.shape[:-1]) + [N_samples])

    #     u = u.contiguous()
    #     inds = torch.searchsorted(cdf, u, right=True)
    #     below = torch.max(torch.zeros_like(inds-1), inds-1)
    #     above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    #     inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    #     matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    #     cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    #     bins_g = torch.gather(rs_mid.unsqueeze(1).expand(matched_shape), 2, inds_g)

    #     denom = cdf_g[...,1] - cdf_g[...,0]
    #     denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    #     t = (u - cdf_g[...,0]) / denom

    #     rs = bins_g[...,0] + t * (bins_g[...,1] - bins_g[...,0])
    #     ts = torch.rand_like(rs) * math.pi
    #     ps = torch.rand_like(rs) * 2 * math.pi

    #     xs = rs * torch.sin(ts) * torch.cos(ps)
    #     ys = rs * torch.sin(ts) * torch.sin(ps)
    #     zs = rs * torch.cos(ts)
    #     samples = torch.cat([xs[...,None], ys[...,None], zs[...,None]], -1)

    #     return samples
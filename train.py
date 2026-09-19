import argparse
import math
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as TF
from torchvision.utils import save_image
from tqdm import tqdm


#===== encoder/decoder (CompVis VQModelInterface / taming VQ) =====
# Names, module tree and forward match the official code so vq-f4 weights load.

def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k) * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)
        return x + self.proj_out(h_)


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "none"], f"attn_type {attn_type} unknown"
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    return nn.Identity()


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False,
                 attn_type="vanilla", **ignore_kwargs):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels,
                                  kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        h = self.norm_out(h)
        h = nonlinearity(h)
        return self.conv_out(h)


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False,
                 use_linear_attn=False, attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class VectorQuantizer(nn.Module):
    """taming VectorQuantizer2 (legacy=True), no einops."""

    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.sane_index_shape = sane_index_shape
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        d = (z_flattened.pow(2).sum(dim=1, keepdim=True)
             + self.embedding.weight.pow(2).sum(dim=1)
             - 2 * (z_flattened @ self.embedding.weight.t()))
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])
        return z_q, loss, (None, None, min_encoding_indices)


class _Dummy:
    """Stand-in for Lightning / OmegaConf objects inside official ckpts."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _IgnoreMissingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(("pytorch_lightning", "lightning", "omegaconf")):
            return _Dummy
        return super().find_class(module, name)


class _ignore_missing_pickle:
    Unpickler = _IgnoreMissingUnpickler
    load = staticmethod(lambda f, **k: _IgnoreMissingUnpickler(f, **k).load())
    dump = pickle.dump
    dumps = pickle.dumps
    loads = pickle.loads


def load_pl_ckpt(path, map_location="cpu"):
    """Load a CompVis Lightning .ckpt without installing pytorch_lightning."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except ModuleNotFoundError:
        return torch.load(
            path, map_location=map_location, weights_only=False,
            pickle_module=_ignore_missing_pickle)


class VQModelInterface(nn.Module):
    """Frozen first stage used by LatentDiffusion. encode() returns pre-quant z."""

    def __init__(self, ddconfig, n_embed, embed_dim, ckpt_path=None, ignore_keys=(),
                 remap=None, sane_index_shape=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=()):
        pl_sd = load_pl_ckpt(path)
        sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
        keys = list(sd.keys())
        ignore_keys = tuple(ignore_keys) + ("loss.",)
        for k in keys:
            if any(k.startswith(ik) for ik in ignore_keys):
                del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"VQ restored from {path}: {len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            print(f"  Missing Keys: {missing}")
        if unexpected:
            print(f"  Unexpected Keys: {unexpected}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        if not force_not_quantize:
            quant, _, _ = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

    def encode_to_prequant(self, x):
        return self.encode(x)

    def decode_code(self, code_b):
        quant_b = self.quantize.embedding(code_b)
        return self.decode(quant_b, force_not_quantize=True)


VQ_F4_DDCONFIG = dict(
    double_z=False,
    z_channels=3,
    resolution=256,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=(1, 2, 4),
    num_res_blocks=2,
    attn_resolutions=[],
    dropout=0.0,
)


def make_vq_f4(ckpt_path=None):
    """Official vq-f4: 256^2 RGB -> 64x64x3 latent, codebook 8192 x 3."""
    return VQModelInterface(
        ddconfig=VQ_F4_DDCONFIG, n_embed=8192, embed_dim=3, ckpt_path=ckpt_path)


#======= diffusion =========

class GaussianDiffusion:
    """Forward process + DDIM reverse. LDM linear schedule (sqrt-space linspace).

    CelebA-HQ LDM-4: linear_start=0.0015, linear_end=0.0195, T=1000.
    """

    def __init__(self, T=1000, beta_start=0.0015, beta_end=0.0195):
        self.T = T
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, T) ** 2
        alphas = 1 - betas
        acp = torch.cumprod(alphas, dim=0)

        self.acp = acp
        self.sqrt_acp = acp.sqrt()
        self.sqrt_1m_acp = (1.0 - acp).sqrt()

    def to(self, device):
        for k, v in vars(self).items():
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self

    def forward_sample(self, x0, t, noise):
        """Closed-form forward: x_t = sqrt(ab_t)*x_0 + sqrt(1-ab_t)*eps."""
        return (extract(self.sqrt_acp, t, x0.shape) * x0 + extract(self.sqrt_1m_acp, t, x0.shape) * noise)

    def predict_x0(self, model, xt, t):
        """Estimate x0_hat from x_t."""
        tv = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
        eps = model(xt, tv)
        x0_hat = (xt - extract(self.sqrt_1m_acp, tv, xt.shape) * eps) \
            / extract(self.sqrt_acp, tv, xt.shape)
        return x0_hat

    def loss(self, model, x0):
        """L_simple: predict the noise eps from x_t."""
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.forward_sample(x0, t, noise)
        return F.mse_loss(model(xt, t), noise)

    @torch.no_grad()
    def ddim_p_sample(self, model, x, t, s, eta=0.0):
        """One DDIM step x_t -> x_s (s < t, arbitrary skip).

        x_s = sqrt(ab_s)*x0_hat + sqrt(1 - ab_s - sigma^2)*eps + sigma*z
        sigma^2 = eta^2 * (1-ab_s)/(1-ab_t) * (1 - ab_t/ab_s)

        eta=0: deterministic DDIM; eta=1: matches DDPM variance.
        """
        tv = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        sv = torch.full((x.shape[0],), s, device=x.device, dtype=torch.long)
        x0_pred = self.predict_x0(model, x, t)
        eps = (x - extract(self.sqrt_acp, tv, x.shape) * x0_pred) \
            / extract(self.sqrt_1m_acp, tv, x.shape)
        acp_t, acp_s = extract(self.acp, tv, x.shape), extract(self.acp, sv, x.shape)
        sigma_var = (eta ** 2) * (1.0 - acp_s) * (1.0 - acp_t / acp_s) / (1.0 - acp_t)
        sigma_var = sigma_var.clamp(min=0.0)
        dir_term = (1.0 - acp_s - sigma_var).sqrt() * eps
        x_s = acp_s.sqrt() * x0_pred + dir_term
        if eta > 0 and s > 0:
            x_s = x_s + sigma_var.sqrt() * torch.randn_like(x)
        return x_s

    def make_ddim_timesteps(self, num_steps, spacing="uniform"):
        """Descending subsequence of original indices in 0..T-1."""
        if num_steps <= 1:
            return [self.T - 1, 0]
        if num_steps >= self.T:
            return list(range(self.T - 1, -1, -1))
        if spacing == "uniform":
            fracs = [i / (num_steps - 1) for i in range(num_steps)]
        elif spacing == "quad":
            fracs = [(i / (num_steps - 1)) ** 2 for i in range(num_steps)]
        else:
            raise ValueError(f"unknown spacing: {spacing!r}")
        return sorted({int(round(f * (self.T - 1))) for f in fracs}, reverse=True)

    @torch.no_grad()
    def ddim_p_sample_loop(self, model, shape, device, num_steps=200, spacing="uniform",
                           eta=0.0, progress_every=None, x_T=None):
        taus = self.make_ddim_timesteps(num_steps, spacing)
        x = torch.randn(shape, device=device) if x_T is None else x_T
        snaps = []
        for i in range(len(taus) - 1):
            t, s = taus[i], taus[i + 1]
            x = self.ddim_p_sample(model, x, t, s, eta=eta)
            if progress_every and (i % progress_every == 0 or i == len(taus) - 2):
                snaps.append(self.predict_x0(model, x, s))
        return x, snaps

#======= U-Net =============

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim, dropout, num_groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)

class SinusoidalTimeEmbed(nn.Module):
    """Transformer sinusoidal embedding of the scalar timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class Unet(nn.Module):
    def __init__(self, in_ch=3, dim=224, mults=(1, 2, 3, 4), dropout=0.0):
        super().__init__()
        d1, d2, d3, d4 = (dim * m for m in mults)
        tdim = dim*4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbed(dim), nn.Linear(dim, tdim), nn.SiLU(), nn.Linear(tdim, tdim)
        )

        self.init_conv = nn.Conv2d(in_ch, d1, 3, padding=1)

        # down path
        self.res1_1 = ResBlock(d1, d1, tdim, dropout)
        self.res1_2 = ResBlock(d1, d1, tdim, dropout)
        self.down1 = nn.Conv2d(d1, d1, 3, stride=2, padding=1)

        self.res2_1 = ResBlock(d1, d2, tdim, dropout)
        self.res2_2 = ResBlock(d2, d2, tdim, dropout)
        self.down2 = nn.Conv2d(d2, d2, 3, stride=2, padding=1)
        self.atten1 = Self_Attention(d2)

        self.res3_1 = ResBlock(d2, d3, tdim, dropout)
        self.res3_2 = ResBlock(d3, d3, tdim, dropout)
        self.down3 = nn.Conv2d(d3, d3, 3, stride=2, padding=1)
        self.atten2 = Self_Attention(d3)

        self.res4_1 = ResBlock(d3, d4, tdim, dropout)
        self.res4_2 = ResBlock(d4, d4, tdim, dropout)
        self.atten3 = Self_Attention(d4)

        # bottleneck
        self.mid1 = ResBlock(d4, d4, tdim, dropout)
        self.mid_attn = Self_Attention(d4)
        self.mid2 = ResBlock(d4, d4, tdim, dropout)

        # up path
        self.res5_1 = ResBlock(d4 * 2, d4, tdim, dropout)
        self.res5_2 = ResBlock(d4, d4, tdim, dropout)
        self.res5_3 = ResBlock(d4, d4, tdim, dropout)
        self.up1 = nn.Conv2d(d4, d4, 3, padding=1)
        self.atten4 = Self_Attention(d4)

        self.res6_1 = ResBlock(d4 + d3, d3, tdim, dropout)
        self.res6_2 = ResBlock(d3, d3, tdim, dropout)
        self.res6_3 = ResBlock(d3, d3, tdim, dropout)
        self.up2 = nn.Conv2d(d3, d3, 3, padding=1)
        self.atten5 = Self_Attention(d3)

        self.res7_1 = ResBlock(d3 + d2, d2, tdim, dropout)
        self.res7_2 = ResBlock(d2, d2, tdim, dropout)
        self.res7_3 = ResBlock(d2, d2, tdim, dropout)        
        self.up3 = nn.Conv2d(d2, d2, 3, padding=1)
        self.atten6 = Self_Attention(d2)

        self.res8_1 = ResBlock(d2 + d1, d1, tdim, dropout)
        self.res8_2 = ResBlock(d1, d1, tdim, dropout)
        self.res8_3 = ResBlock(d1, d1, tdim, dropout)        

        # output
        self.out_norm = nn.GroupNorm(8, d1)
        self.out_conv = nn.Conv2d(d1, in_ch, 3, padding=1)


    def forward(self, x, t):
        temb = self.time_mlp(t)
        h = self.init_conv(x)
        
        s0 = self.res1_2(self.res1_1(h, temb), temb)
        h = self.down1(s0)
        s1 = self.atten1(self.res2_2(self.res2_1(h, temb), temb))
        h = self.down2(s1)
        s2 = self.atten2(self.res3_2(self.res3_1(h, temb), temb))
        h = self.down3(s2)
        s3 = self.atten3(self.res4_2(self.res4_1(h, temb), temb))
        
        h = self.mid2(self.mid_attn(self.mid1(s3, temb)), temb)

        h = self.res5_1(torch.cat([h, s3], dim=1), temb)
        h = self.atten4(h)
        h = self.res5_3(self.res5_2(h, temb), temb)
        h = self.up1(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = self.res6_1(torch.cat([h, s2], dim=1), temb)
        h = self.atten5(h)
        h = self.res6_3(self.res6_2(h, temb), temb)
        h = self.up2(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = self.res7_1(torch.cat([h, s1], dim=1), temb)
        h = self.atten6(h)
        h = self.res7_3(self.res7_2(h, temb), temb)
        h = self.up3(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = self.res8_1(torch.cat([h, s0], dim=1), temb)
        h = self.res8_3(self.res8_2(h, temb), temb)

        return self.out_conv(F.silu(self.out_norm(h)))

class Self_Attention(nn.Module):
    def __init__(self, ch, head_channels=32, num_groups=32):
        super().__init__()
        self.num_heads = ch // head_channels
        self.norm = nn.GroupNorm(num_groups, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).view(B, 3, self.num_heads, C//self.num_heads, H*W)
        q, k, v = qkv.permute(1, 0, 2, 4, 3).unbind(0)
        att_wei = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        y = (att_wei.softmax(dim=-1) @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(y)

#======= tools =============

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _is_imagefolder(root):
    """True if root/class_name/ contains at least one image."""
    if not os.path.isdir(root):
        return False
    for cls in os.listdir(root):
        p = os.path.join(root, cls)
        if not os.path.isdir(p) or cls.startswith("."):
            continue
        try:
            names = os.listdir(p)
        except OSError:
            continue
        if any(f.lower().endswith(_IMG_EXTS) for f in names):
            return True
    return False


def ensure_celebahq(data_dir):
    """Download CelebA-HQ 256^2 (30k) into data_dir/celebahq/all/ if needed."""
    root = os.path.join(data_dir, "celebahq")
    img_dir = os.path.join(root, "all")
    marker = os.path.join(root, ".done")
    if os.path.exists(marker) and _is_imagefolder(root):
        return root

    os.makedirs(img_dir, exist_ok=True)
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "CelebA-HQ auto-download requires the `datasets` package: pip install datasets"
        ) from e

    print("Downloading CelebA-HQ 256^2 from Hugging Face (korexyz/celeba-hq-256x256)...")
    ds = load_dataset("korexyz/celeba-hq-256x256")
    idx = 0
    for split in ("train", "validation"):
        if split not in ds:
            continue
        for row in tqdm(ds[split], desc=f"export CelebA-HQ {split}"):
            out = os.path.join(img_dir, f"{idx:05d}.jpg")
            if not os.path.exists(out):
                img = row["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(out, quality=95)
            idx += 1
    with open(marker, "w", encoding="utf-8") as f:
        f.write(f"{idx}\n")
    print(f"CelebA-HQ ready: {img_dir} ({idx} images)")
    return root


def get_loader(data_dir, batch_size, size=256):
    """256^2 images in [-1, 1]. ImageFolder if present, else auto-download CelebA-HQ."""

    tf = TF.Compose([
        TF.Resize(size),
        TF.CenterCrop(size),
        TF.RandomHorizontalFlip(),
        TF.ToTensor(),
        TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if _is_imagefolder(data_dir):
        root = data_dir
        print(f"ImageFolder: {data_dir}")
    else:
        root = ensure_celebahq(data_dir)
    ds = torchvision.datasets.ImageFolder(root, transform=tf)
    print(f"{len(ds)} images from {root}")
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2,
                    drop_last=True, pin_memory=True, persistent_workers=True)


def load_frozen_vq(ckpt_path, device):
    """Load official vq-f4 and freeze; diffusion never trains these weights."""
    vq = make_vq_f4(ckpt_path).to(device)
    vq.eval()
    for p in vq.parameters():
        p.requires_grad = False
    n = sum(p.numel() for p in vq.parameters())
    print(f"VQ-f4 frozen: {n / 1e6:.2f}M from {ckpt_path}")
    return vq


@torch.no_grad()
def decode_to_image(vq, z):
    """Latent -> RGB in [0, 1]."""
    return ((vq.decode(z) + 1) / 2).clamp(0, 1)

class EMA:
    """Exponential moving average of model weights (paper: decay 0.9999)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

def extract(a, t, x_shape):
    """Gather from precomputed tensor a[T] at timesteps t[B], reshape for broadcasting."""
    return a.gather(0, t).view(-1, *([1] * (len(x_shape) - 1)))


@torch.no_grad()
def run_with_ema(model, ema, fn):
    """Swap EMA weights in-place (train weights parked on CPU) so sampling does not clone the U-Net on GPU."""
    device = next(model.parameters()).device
    backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema.shadow)
    model.eval()
    try:
        return fn(model)
    finally:
        model.load_state_dict({k: v.to(device) for k, v in backup.items()})
        model.train()


def ensure_fid_ref(data_dir):
    """FID reference = training images on disk (no flip). ImageFolder or auto CelebA-HQ."""
    if _is_imagefolder(data_dir):
        return data_dir
    return ensure_celebahq(data_dir)


@torch.no_grad()
def generate_fid_samples(model, vq, diffusion, device, out_dir, n, batch_size,
                         ddim_steps, spacing, eta, latent_size, in_ch):
    """DDIM in latent space, VQ-decode, save PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):
        if old.endswith(".png"):
            os.remove(os.path.join(out_dir, old))
    idx = 0
    pbar = tqdm(total=n, desc=f"generate FID samples (ddim {ddim_steps})")
    while idx < n:
        bs = min(batch_size, n - idx)
        z, _ = diffusion.ddim_p_sample_loop(
            model, (bs, in_ch, latent_size, latent_size), device,
            num_steps=ddim_steps, spacing=spacing, eta=eta)
        imgs = decode_to_image(vq, z)
        for j in range(bs):
            save_image(imgs[j], os.path.join(out_dir, f"{idx + j:05d}.png"))
        idx += bs
        pbar.update(bs)
    pbar.close()


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Fréchet distance between two Gaussians (scipy>=1.14 compatible)."""
    import numpy as np
    from scipy import linalg

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2

    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        print(f"FID: singular product; adding {eps} to diagonal of cov estimates")
        eye = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + eye) @ (sigma2 + eye))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def _list_images(img_dir):
    """Collect image paths under img_dir (class subdirs allowed)."""
    files = []
    for dirpath, _, names in os.walk(img_dir):
        for f in names:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                files.append(os.path.join(dirpath, f))
    files.sort()
    if not files:
        raise FileNotFoundError(f"no images found in {img_dir}")
    return files


def compute_fid(fake_dir, ref_dir, device, batch_size=50):
    """Extract Inception features with pytorch-fid, Fréchet dist with our helper."""
    from pytorch_fid import fid_score
    from pytorch_fid.inception import InceptionV3

    fake_files = _list_images(fake_dir)
    ref_files = _list_images(ref_dir)
    print(f"FID: {len(fake_files)} fake vs {len(ref_files)} ref images")

    dims = 2048
    block = max(1, min(batch_size, 256))
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]]).to(device)
    model.eval()

    mu_g, sigma_g = fid_score.calculate_activation_statistics(
        fake_files, model, block, dims, device, num_workers=0)
    mu_r, sigma_r = fid_score.calculate_activation_statistics(
        ref_files, model, block, dims, device, num_workers=0)
    return _frechet_distance(mu_g, sigma_g, mu_r, sigma_r)


def run_fid_eval(model, ema, vq, diffusion, device, args, step=None, writer=None):
    """Generate fakes with EMA + DDIM, FID vs the training image folder."""
    ref_dir = ensure_fid_ref(args.data_dir)
    fake_dir = os.path.join(args.out, "fid", "fake")
    t0 = time.time()

    def _gen(m):
        generate_fid_samples(
            m, vq, diffusion, device, fake_dir, args.n_fid, args.fid_batch,
            args.ddim_steps, args.ddim_spacing, args.eta,
            args.latent_size, args.in_ch)

    if ema is not None:
        run_with_ema(model, ema, _gen)
    else:
        model.eval()
        _gen(model)
    fid = compute_fid(fake_dir, ref_dir, device, batch_size=args.fid_batch)
    dt = time.time() - t0
    tag = f"step {step}" if step is not None else "final"
    print(f"FID ({tag}, n={args.n_fid}, ddim {args.ddim_steps}): {fid:.2f}  ({dt / 60:.1f} min)")
    if writer is not None and step is not None:
        writer.add_scalar("eval/fid", fid, step)
    return fid

#======= train & sample ====

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    assert args.vq_ckpt, "training requires --vq_ckpt (official vq-f4 model.ckpt)"

    vq = load_frozen_vq(args.vq_ckpt, device)
    model = Unet(in_ch=args.in_ch, dim=args.dim, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net params: {n_params / 1e6:.2f}M, device: {device}")

    diffusion = GaussianDiffusion(
        T=args.T, beta_start=args.beta_start, beta_end=args.beta_end).to(device)
    ema = EMA(model, decay=args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step, losses = 0, []
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        ema.shadow = ckpt["ema"]
        step, losses = ckpt["step"], ckpt.get("losses", [])
        print(f"resumed from checkpoint: step {step}")

    loader = get_loader(args.data_dir, args.batch_size, size=args.image_size)
    tb_dir = os.path.join(args.out, "tb")
    writer = SummaryWriter(log_dir=tb_dir)
    writer.add_text("config", "\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())), 0)
    print(f"tensorboard: tensorboard --logdir {tb_dir}")

    def save_ckpt():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "ema": ema.shadow, "step": step, "losses": losses,
                    "args": vars(args)}, os.path.join(args.out, "ckpt.pt"))

    sample_noise = torch.randn(
        args.n_samples, args.in_ch, args.latent_size, args.latent_size, device=device)

    def quick_sample(tag):
        """Fixed latent noise so samples_{step}.png is comparable across training."""

        def _sample(m):
            z, _ = diffusion.ddim_p_sample_loop(
                m, sample_noise.shape, device,
                num_steps=args.ddim_steps, spacing=args.ddim_spacing, eta=args.eta,
                x_T=sample_noise)
            return decode_to_image(vq, z)

        imgs = run_with_ema(model, ema, _sample)
        save_image(imgs, os.path.join(args.out, f"samples_{tag}.png"),
                   nrow=4, value_range=(0, 1))
        writer.add_images("samples/grid", imgs, global_step=tag)

    t0 = time.time()
    while step < args.max_steps:
        for x, _ in loader:
            if step >= args.max_steps:
                break
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                z = vq.encode(x)
            loss = diffusion.loss(model, z)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(model)
            step += 1

            if step % args.log_every == 0:
                losses.append(loss.item())
                dt = (time.time() - t0) / args.log_every * 1000
                t0 = time.time()
                print(f"step {step:>7d} | loss {loss.item():.4f} | {dt:.0f} ms/step")
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/ms_per_step", dt, step)

            if step % args.sample_every == 0:
                quick_sample(step)
                print(f"    saved samples_{step}.png")

            if args.fid_every > 0 and step % args.fid_every == 0:
                run_fid_eval(model, ema, vq, diffusion, device, args, step=step, writer=writer)

            if step % args.ckpt_every == 0:
                save_ckpt()

    if step % args.sample_every != 0:
        quick_sample(step)
        print(f"    saved samples_{step}.png")
    save_ckpt()
    if args.fid_final:
        run_fid_eval(model, ema, vq, diffusion, device, args, step=step, writer=writer)
    writer.close()
    print(f"training done, {step} steps, checkpoint: {os.path.join(args.out, 'ckpt.pt')}")


def sample(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ca = ckpt["args"]
    vq_path = args.vq_ckpt or ca.get("vq_ckpt")
    assert vq_path, "sampling requires --vq_ckpt or a checkpoint saved with vq_ckpt"

    vq = load_frozen_vq(vq_path, device)
    model = Unet(in_ch=ca["in_ch"], dim=ca["dim"], dropout=ca["dropout"]).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()

    diffusion = GaussianDiffusion(
        T=ca["T"], beta_start=ca["beta_start"], beta_end=ca["beta_end"]).to(device)
    z_size = ca["latent_size"]
    z, snaps = diffusion.ddim_p_sample_loop(
        model, (args.n_samples, ca["in_ch"], z_size, z_size), device,
        num_steps=args.ddim_steps, spacing=args.ddim_spacing, eta=args.eta,
        progress_every=args.progress_every)
    imgs = decode_to_image(vq, z)
    out = os.path.join(args.out, "samples_final.png")
    save_image(imgs, out, nrow=4, value_range=(0, 1))
    print(f"samples: {out}")

    if snaps:
        recs = torch.cat([decode_to_image(vq, s) for s in snaps], dim=0)
        out_p = os.path.join(args.out, "progression.png")
        save_image(recs, out_p, nrow=args.n_samples, value_range=(0, 1))
        print(f"progression: {out_p} (rows: decoded x0_hat along DDIM, noisy -> clean)")


def eval_fid(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ca = ckpt["args"]
    vq_path = args.vq_ckpt or ca.get("vq_ckpt")
    assert vq_path, "FID requires --vq_ckpt or a checkpoint saved with vq_ckpt"

    vq = load_frozen_vq(vq_path, device)
    model = Unet(in_ch=ca["in_ch"], dim=ca["dim"], dropout=ca["dropout"]).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    diffusion = GaussianDiffusion(
        T=ca["T"], beta_start=ca["beta_start"], beta_end=ca["beta_end"]).to(device)

    fid_args = argparse.Namespace(
        data_dir=args.data_dir,
        out=args.out,
        n_fid=args.n_fid,
        fid_batch=args.fid_batch,
        ddim_steps=args.ddim_steps,
        ddim_spacing=args.ddim_spacing,
        eta=args.eta,
        latent_size=ca["latent_size"],
        in_ch=ca["in_ch"],
    )
    run_fid_eval(model, None, vq, diffusion, device, fid_args)


#======= main ==============

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Unconditional LDM (latent U-Net)")
    # training  (CelebA-HQ LDM-4 Table 12)
    p.add_argument("--max_steps", type=int, default=410_000)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--lr", type=float, default=9.6e-5)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--resume", type=str, default=None)
    # diffusion / U-Net
    p.add_argument("--in_ch", type=int, default=3)
    p.add_argument("--dim", type=int, default=224)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--latent_size", type=int, default=64)
    p.add_argument("--vq_ckpt", type=str, default=None,
                   help="official vq-f4 Lightning ckpt (required for train/sample)")
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=0.0015)
    p.add_argument("--beta_end", type=float, default=0.0195)
    # sampling (CelebA-HQ LDM-4: 200 DDIM steps, eta=0)
    p.add_argument("--ddim_steps", type=int, default=200)
    p.add_argument("--ddim_spacing", type=str, default="uniform", choices=["uniform", "quad"])
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--sample", action="store_true", help="sample only (requires --ckpt)")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--progress_every", type=int, default=10,
                   help="DDIM-step interval for progression.png (only used with --sample)")
    # FID (paper: 50k samples vs CelebA-HQ train images, 200-step DDIM)
    p.add_argument("--eval_fid", action="store_true", help="FID only (requires --ckpt)")
    p.add_argument("--fid_every", type=int, default=0,
                   help="FID every N training steps (0=disable)")
    p.add_argument("--fid_final", action="store_true",
                   help="run FID once after training finishes")
    p.add_argument("--n_fid", type=int, default=10_000,
                   help="generated images for FID")
    p.add_argument("--fid_batch", type=int, default=8,
                   help="batch size for FID generation / Inception")
    # misc
    p.add_argument("--data_dir", type=str, default="data",
                   help="ImageFolder root, or cache dir for auto-downloaded CelebA-HQ")
    p.add_argument("--out", type=str, default="runs/ldm_uncond")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--sample_every", type=int, default=5000)
    p.add_argument("--ckpt_every", type=int, default=5000)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.eval_fid:
        assert args.ckpt, "--eval_fid requires --ckpt"
        eval_fid(args)
    elif args.sample:
        assert args.ckpt, "--sample requires --ckpt"
        sample(args)
    else:
        train(args)


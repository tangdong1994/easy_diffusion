import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import nn, einsum
from einops import rearrange, repeat
import math
from abc import abstractmethod
from inspect import isfunction

_ATTN_PRECISION = "fp32"


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=(1, 2, 3, 6), pool_type="average", activation=nn.LeakyReLU):
        super(PSPModule, self).__init__()
        
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        if pool_type == "average":
            pool = nn.AdaptiveAvgPool2d
        else:
            pool = nn.AdaptiveMaxPool2d
            
        self.modlist = nn.ModuleList()

        for pool_size in pool_sizes:
            self.modlist.append(
                nn.Sequential(
                    pool(pool_size),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    activation()
                )
            )
        self.out = nn.Conv2d(in_channels+len(pool_sizes)*out_channels, in_channels, kernel_size=3, padding=1)
        

    def forward(self, x):
        feats = []
        feats.append(x)
        for ppm in self.modlist:
            feat = F.interpolate(ppm(x), size=x.size()[2:], mode='bilinear', align_corners=True)
            feats.append(feat)

        out = torch.cat(feats, dim=1)
        out = self.out(out)
        out = F.relu(out)

        return out
    
    
class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)
    
    
def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")
    
    
def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")
    
    
class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResBlock2(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock2(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResBlock2(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock2(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResBlock2(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResBlock2(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
    
    
def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class ResBlock2(TimestepBlock):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]
        #h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

    
    
class ResBlock(TimestepBlock):
    """skip connection and time embeding
    """
    def __init__(self, in_channels, embed_channels, dropout, out_channels=None, use_conv=False, ways="identity", dims=2):
        super().__init__()
        assert ways in ['identity','up','down'], 'please select ways in [identity,up,down]'
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.out_channels = out_channels or in_channels
        self.ways = ways
        
        if ways=="identity":
            self.h_func = self.x_func = conv_nd(dims, in_channels, self.out_channels, 3, padding=1)#nn.Identity()
        elif ways == "up":
            self.h_func = Upsample(in_channels, True, dims)
            self.x_func = Upsample(in_channels, True, dims)
        else:
            self.h_func = Downsample(in_channels, True, dims)
            self.x_func = Downsample(in_channels, True, dims)
            
        self.in_layers = nn.Sequential(
            normalization(in_channels),
            nn.SiLU(),
            conv_nd(dims, in_channels, self.out_channels, 3, padding=1),
        )
        
        self.time_embe = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_channels, self.out_channels)
        )
        
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout)
        )
        
        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, in_channels, self.out_channels, 1)
    def forward(self, x, emb):
        if self.ways!="identity":
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_func(h)
            x = self.x_func(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.time_embe(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h
    

class MultiCrossAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, context_dim=None, num_heads=1,):
        """
        Note: For simplicity reason, we just implemented 1-head attention. 
        Feel free to implement multi-head attention! with fancy tensor manipulations.
        """
        super(MultiCrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.each_dim = embed_dim//num_heads
        self.num_heads = num_heads
        self.query = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.cancat = nn.Linear(hidden_dim, hidden_dim, bias=False)
        if context_dim is None:
            self.self_attn = True 
            self.key = nn.Linear(hidden_dim, embed_dim, bias=False)     ###########
            self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)  ############
        else:
            self.self_attn = False 
            self.key = nn.Linear(context_dim, embed_dim, bias=False)   #############
            self.value = nn.Linear(context_dim, hidden_dim, bias=False) ############

    
    def forward(self, tokens, context=None):
        # tokens: with shape [batch, sequence_len, hidden_dim]
        # context: with shape [batch, contex_seq_len, context_dim]
        batch_size, seq_length, _ = tokens.size()
        seq_length2 = seq_length
        if self.self_attn:
            Q = self.query(tokens)
            K = self.key(tokens)  
            V = self.value(tokens) 
        else:
            # print("MultiCrossAttention context:",context.shape)
            _, seq_length2, _ = context.size()

            # implement Q, K, V for the Cross attention 
            Q = self.query(tokens)
            K = self.key(context)
            V = self.value(context)
        
        #print("before Q:", Q.shape)
        Q = Q.reshape(batch_size, seq_length, self.num_heads, -1).permute(0, 2, 1, 3)
        #print("after Q:",Q.shape)
        #print("before K:", K.shape)
        K = K.reshape(batch_size, seq_length2, self.num_heads, -1).permute(0, 2, 1, 3)
        #print("after K:", K.shape)
        #print("before V:",V.shape)
        V = V.reshape(batch_size, seq_length2, self.num_heads, -1).permute(0, 2, 1, 3)
        #print("after V:",V.shape)
        ####### YOUR CODE HERE (2 lines)\
        scoremats = torch.einsum("bnij,bnkj->bnik" ,Q, K)/math.sqrt(Q.size()[-1])         # inner product of Q and K, a tensor 
        attnmats = F.softmax(scoremats, dim=-1)          # softmax of scoremats
        # print(scoremats.shape, attnmats.shape, )
        # print("before V:",V.shape,V)
        ctx_vecs = torch.einsum("bnts,bnsh->bnth", attnmats, V)  # weighted average value vectors by attnmats
        # print("after V:",ctx_vecs.shape,ctx_vecs)
        # print(ctx_vecs.shape)
        ctx_vecs = ctx_vecs.permute(0, 2, 1, 3).reshape(batch_size, seq_length, -1)
        # print(ctx_vecs.shape)
        ctx_vecs = self.cancat(ctx_vecs)
        return ctx_vecs


class MultiCrossAttentionV2(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, num_heads=64, dropout=0.):
        super().__init__()
        inner_dim = num_heads * heads
        context_dim = default(context_dim, query_dim)

        self.scale = num_heads ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        # if _ATTN_PRECISION =="fp32":
        #    with torch.cuda.amp.autocast():
        #        q, k = q.float(), k.float()
        #        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # else:
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)
        # print("before v:",v.shape,v)
        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

    
class TransformerBlock(nn.Module):
    """The transformer block that combines self-attn, cross-attn and feed forward neural net"""
    def __init__(self, hidden_dim, context_dim, num_heads=1):
        super(TransformerBlock, self).__init__()
        self.attn_self = MultiCrossAttention(hidden_dim, hidden_dim, num_heads=num_heads)
        self.attn_cross = MultiCrossAttention(hidden_dim, hidden_dim, context_dim, num_heads=num_heads)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        # implement a 2 layer MLP with K*hidden_dim hidden units, and nn.GeLU nonlinearity #######
        self.ffn  = nn.Sequential(
            # YOUR CODE HERE ##################
            nn.Linear(hidden_dim, 3*hidden_dim),
            nn.GELU(),
            nn.Linear(3*hidden_dim, hidden_dim)
        )
    
    def forward(self, x, context=None):
        # print("TransformerBlock2 context:",context.shape)
        # Notice the + x as residue connections
        x = self.attn_self(self.norm1(x)) + x
        # Notice the + x as residue connections
        x = self.attn_cross(self.norm2(x), context=context) + x
        # Notice the + x as residue connections
        x = self.ffn(self.norm3(x)) + x
        return x     
    

class TransformerBlockV2(nn.Module):
    """The transformer block that combines self-attn, cross-attn and feed forward neural net"""
    def __init__(self, hidden_dim, context_dim, heads, num_heads=1):
        super(TransformerBlockV2, self).__init__()
        self.attn_self = MultiCrossAttentionV2(hidden_dim, heads=heads, num_heads=num_heads)
        self.attn_cross = MultiCrossAttentionV2(hidden_dim, context_dim, heads=heads, num_heads=num_heads)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        # implement a 2 layer MLP with K*hidden_dim hidden units, and nn.GeLU nonlinearity #######
        self.ffn  = nn.Sequential(
            # YOUR CODE HERE ##################
            nn.Linear(hidden_dim, 3*hidden_dim),
            nn.GELU(),
            nn.Linear(3*hidden_dim, hidden_dim)
        )
    
    def forward(self, x, context=None):
        # print("TransformerBlock2 context:",context.shape)
        # Notice the + x as residue connections
        x = self.attn_self(self.norm1(x)) + x
        # Notice the + x as residue connections
        x = self.attn_cross(self.norm2(x), context=context) + x
        # Notice the + x as residue connections
        x = self.ffn(self.norm3(x)) + x
        return x 

    
class SpatialTransformer(nn.Module):
    def __init__(self, hidden_dim, context_dim, num_heads=1):
        super(SpatialTransformer, self).__init__()
        self.transformer = TransformerBlock(hidden_dim, context_dim, num_heads=num_heads)

    def forward(self, x, context=None):
        # print("SpatialTransformer2 context:", context.shape)
        b, c, h, w = x.shape
        x_in = x
        # Combine the spatial dimensions and move the channel dimen to the end
        x = rearrange(x, "b c h w->b (h w) c")
        # Apply the sequence transformer
        x = self.transformer(x, context)
        # Reverse the process
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        # Residue 
        return x + x_in
    

class SpatialTransformerV2(nn.Module):
    def __init__(self, hidden_dim, context_dim, heads, num_heads=1):
        super(SpatialTransformerV2, self).__init__()
        self.transformer = TransformerBlockV2(hidden_dim, context_dim, heads=heads, num_heads=num_heads)

    def forward(self, x, context=None):
        # print("SpatialTransformer2 context:", context.shape)
        b, c, h, w = x.shape
        x_in = x
        # Combine the spatial dimensions and move the channel dimen to the end
        x = rearrange(x, "b c h w->b (h w) c")
        # Apply the sequence transformer
        x = self.transformer(x, context)
        # Reverse the process
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        # Residue 
        return x + x_in
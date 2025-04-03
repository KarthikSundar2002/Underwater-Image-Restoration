import math
from re import S
import token
from networkx import from_prufer_sequence
from numpy import short
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_, to_2tuple, DropPath

from model.wave_modules import DWT_2D, IDWT_2D

def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

class InputProjection(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU()):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True),
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = None
        self.in_channel = in_channels
        self.out_channel = out_channels

    def forward(self, x):
        B,C,H,W = x.shape
        x = self.proj(x).flatten(2).transpose(1,2).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x
    
class OutputProjection(nn.Module):
    def __init__(self, in_channels=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.act = act_layer(inplace=True)
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channels
        self.out_channel = out_channel
    
    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()
        out = self.proj(x)
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out
    
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
        self.in_channel = in_channel
        self.out_channel = out_channel
    
    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()
        return out
    
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()
        return out



class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())

        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1), act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = nn.Identity()

    def forward(self, x):
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=hh, w=hh)

        x = self.dwconv(x)

        x = rearrange(x, 'b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear2(x)
        x = self.eca(x)
        return x

class FRFN(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim*2), act_layer())
        self.dwconv = nn.Sequential((nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1), act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, padding=1, stride=1, bias=False)

    def forward(self, x):
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = rearrange(x, 'b (h w) c -> b c h w', h=hh, w=hh)

        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), dim=1)

        x = rearrange(x, 'b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear1(x)
        x_1, x_2 = x.chunk(2, dim=-1)

        x_1 = rearrange(x_1, 'b (h w) c -> b c h w', h=hh, w=hh)
        x_1 = self.dwconv(x_1)
        x_1 = rearrange(x_1, 'b c h w -> b (h w) c', h=hh, w=hh)
        x = x_1 * x_2

        x = self.linear2(x)
        return x

class WindowAttention_Sparse(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative Positional Encoding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads)
        )


        coords_h = torch.arange(self.win_size[0])
        coords_w = torch.arange(self.win_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1,2,0).contiguous()
        relative_coords[:, :, 0] += self.win_size[0] - 1
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # Truncation Normal Initialization - Fills the Input Tensor
        # with values from a normal distribution with mean 0 and std 0.02
        # The values are also initialized within bounds
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection == 'linear':
            self.to_qkv = LinearProjection(dim, num_heads, head_dim, qkv_bias)
        else:
            raise Exception("Projection Error!")
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2))
    
    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q,k,v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1
        ) # Wh * Ww, Wh * Ww, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # num_heads, Wh * Ww, Wh * Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn)**2 # b,h,w,c
        else:
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn)**2
        
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attn = attn0 * w1 + attn1 * w2
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, 

    def extra_repr(self):
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

    
class MDASSA(nn.Module):
    def __init__(self, dim, win_size ,num_heads, qk_scale=None, qkv_bias=True, token_projection='linear', attn_drop=0., proj_drop=0.):    
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_Sparse(
            dim, win_size=to_2tuple(win_size),
            num_heads=num_heads, token_projection=token_projection, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        
        self.conv1x1 = nn.Conv2d(dim, dim*2, kernel_size=1, stride=1, padding=0)
        self.fdfp = FDFP(dim, dim*2, act_layer=act_layer)
        self.freq_attn = WindowAttention_Sparse(
            dim*2, win_size=to_2tuple(win_size),
            num_heads=num_heads, token_projection=token_projection, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        
        self.spatial_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.freq_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x, mask=None):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))

        if mask != None:
            input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
            input_mask_windows = window_partition(input_mask, self.win_size)
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)
            attn_mask = attn_mask.unsqueeze(2)*attn_mask.unsqueeze(1)
            attn_mask = attn_mask.torch.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        if self.shift_size > 0:
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size), slice(-self.win_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size), slice(-self.win_size, -self.shift_size), slice(-self.shift_size, None))

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2)
            shift_attn_mask = shift_attn_mask.torch.masked_fill(shift_attn_mask!=0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask

        shortcut = x 
        freq_in = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        x_windows = window_partition(shifted_x, self.win_size)
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, (H, W))

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        x = shortcut + self.spatial_drop_path(x)

        freq_q = self.fdfp(freq_in)
        kv = self.conv1x1(x)
        
        freq_attn = self.freq_attn(freq_q, attn_kv=kv, mask=mask)
        freq_attn = x + self.freq_drop_path(freq_attn)

        return freq_attn
    

class FDFP(nn.Module):
    def __init__(self, in_channels, hidden_channels,act_layer=nn.GELU):
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1, padding=1)
        self.act = act_layer()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
    
    def forward(self, x):
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.dwt(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.idwt(x)
        x = rearrange(x, 'b c h w -> b h w c')
        
        return x
    
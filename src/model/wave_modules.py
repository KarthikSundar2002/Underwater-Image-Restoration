
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

from einops import rearrange

class DWT_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        
        # print(f"Input shape: {x.shape}")
        x_ll = F.conv2d(x, w_ll, stride = 2)
        x_lh = F.conv2d(x, w_lh, stride = 2)
        x_hl = F.conv2d(x, w_hl, stride = 2)
        x_hh = F.conv2d(x, w_hh, stride = 2)
        x = torch.cat((x_ll, x_lh, x_hl, x_hh), dim=1)
        
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B,C,H,W = ctx.shape
            dx = dx.view(B, 4, -1, H//2, W//2)

            dx = dx.transpose(1,2).reshape(B, -1, H//2, W//2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=1)

        return dx, None, None, None, None

class IDWT_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape
        

        B, C, H, W = x.shape
        # print(f"IDWT input shape: {x.shape}")
        x = rearrange(x, 'b (n c) h w -> b c n h w', n = 4)
        
        C = x.shape[1]
        x = rearrange(x, 'b c n h w -> b (n c) h w')
        
        dim = x.shape[1]
        filters = filters.expand(dim, 4, 2, 2)

        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        #print(f"IDWT output shape: {x.shape}")
        return x
    
    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()
            C = dx.shape[1]
            dx = dx.reshape(B, -1, H//2, W//2)
            filters = filters.expand(C, 1, 1, 1)
            dx = torch.nn.functional.conv2d(dx, filters, stride=2, groups=C)
        return dx, None

class DWT_2D(nn.Module):
    def __init__(self, wave='haar'):
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(wavelet.dec_hi[::-1])
        dec_lo = torch.Tensor(wavelet.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll)
        self.register_buffer('w_hl', w_hl)
        self.register_buffer('w_lh', w_lh)
        self.register_buffer('w_hh', w_hh)

        self.w_ll = w_ll
        self.w_lh = w_lh
        self.w_hl = w_hl
        self.w_hh = w_hh

    def forward(self, x):
        a = self.w_ll.shape[0]
        b = self.w_ll.shape[1]
        dim = x.shape[1]
        self.w_ll = self.w_ll.expand(dim//4, dim, a, b)
        self.w_lh = self.w_lh.expand(dim//4, dim, a, b)
        self.w_hl = self.w_hl.expand(dim//4, dim, a, b)
        self.w_hh = self.w_hh.expand(dim//4, dim, a, b)
        # print(f"w_ll shape: {self.w_ll.shape}")
        # print(f"w_lh shape: {self.w_lh.shape}")
        # print(f"w_hl shape: {self.w_hl.shape}")
        # print(f"w_hh shape: {self.w_hh.shape}")


        return DWT_function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

class IDWT_2D(nn.Module):
    def __init__(self, wave='haar'):
        super(IDWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(wavelet.rec_hi)
        rec_lo = torch.Tensor(wavelet.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        # w_ll = w_ll.
        # w_hl = w_hl.unsqueeze(0).unsqueeze(0)
        # w_lh = w_lh.unsqueeze(0).unsqueeze(0)
        # w_hh = w_hh

        filters = torch.stack([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)
    
    def forward(self, x):
        # print(f"IDWT input shape: {x.shape}")
        # print(f"IDWT filters shape: {self.filters.shape}")
        return IDWT_function.apply(x, self.filters)

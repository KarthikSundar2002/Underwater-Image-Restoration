import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import torchvision.transforms.functional

from timm.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

from src.model.block import FRFN, Downsample,Upsample, InputProjection, OutputProjection, LeFF, MDASSA
from src.model.wave_modules import DWT_2D, IDWT_2D

import math


class EncoderBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads,
                 mlp_ratio=4,token_mlp="leff",drop_path=0.0, norm_layer=nn.LayerNorm
                 , act_layer=nn.GELU, drop=0.0, freq_mlp="leff", use_dwt="Fourier"):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.use_dwt = use_dwt

        mlp_hidden_dim = int(dim * mlp_ratio)

        if token_mlp == "leff":
            self.mlp = LeFF(dim, mlp_hidden_dim,act_layer=act_layer, drop=drop)
        elif token_mlp == "frfn":
            self.mlp = FRFN(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            raise ValueError(f"Unknown token_mlp type: {token_mlp}")
        
        if self.use_dwt == "Wavelet":
            self.dwt = DWT_2D()
        self.norm2 = norm_layer(dim)
        
        if freq_mlp == "leff":
            self.freq_mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif freq_mlp == "frfn":
            self.freq_mlp = FRFN(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            raise ValueError(f"Unknown freq_mlp type: {freq_mlp}")
        
        if self.use_dwt == "Wavelet":
            self.idwt = IDWT_2D()
        
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x, mask=None):
        B, L, C = x.shape
        H, W = int(math.sqrt(L)), int(math.sqrt(L))

        shortcut = x
        freq_x = self.norm2(x)
        x = self.norm1(x)     
        x = self.mlp(x)

     
        freq_x = rearrange(freq_x,'b (h w) c -> b c h w', h=H, w=W)
        if self.use_dwt == "Wavelet":
            freq_x = self.dwt(freq_x)
            freq_x = rearrange(freq_x, 'b c h w -> b (h w) c')
        elif self.use_dwt == "Fourier":
            freq_x = fft.fftn(x, dim=(-2, -1)).real
        else:
            freq_x = freq_x
        # freq_x = self.dwt(freq_x)
   

        freq_x = self.freq_mlp(freq_x)
    

        if self.use_dwt == "Wavelet":
            freq_x = rearrange(freq_x, 'b (h w) c -> b c h w', h=H // 2, w=W // 2)
            freq_x = self.idwt(freq_x)
            freq_x = rearrange(freq_x, 'b c h w -> b (h w) c')
        elif self.use_dwt == "Fourier":
            freq_x = fft.ifftn(freq_x, dim=(-2, -1)).real
        else:
            freq_x = freq_x
       

        x = shortcut + self.drop_path2(freq_x) +self.drop_path(x)
       # x = shortcut + freq_x + x
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads,win_size=8, shift_size=0,
                 mlp_ratio=4,token_mlp="leff",drop_path=0.0, norm_layer=nn.LayerNorm
                 , act_layer=nn.GELU, drop=0.0, token_projection="linear", enc_out=True, freq_attn_win_ratio=2, use_dwt=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        self.enc_out = enc_out
        self.win_size = win_size
        self.shift_size = shift_size

        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        
        assert 0 <= self.shift_size < self.win_size

       
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        mdssa_dim = dim*2 if enc_out else dim
        if self.enc_out:
            self.norm1 = norm_layer(dim*2)
            self.norm2 = norm_layer(dim*2)
        else:
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)

        self.mdassa = MDASSA(mdssa_dim, num_heads=num_heads, win_size=win_size, shift_size=shift_size,
                            token_projection=token_projection, qkv_bias=True, qk_scale=None,
                             attn_drop=0., proj_drop=0., norm_layer=norm_layer, enc_out=enc_out, freq_attn_win_ratio=freq_attn_win_ratio, use_dwt=use_dwt)
        
        mlp_hidden_dim = int(mdssa_dim * mlp_ratio)
        if token_mlp == "leff":
            self.mlp = LeFF(mdssa_dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == "frfn":
            self.mlp = FRFN(mdssa_dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            raise ValueError(f"Unknown token_mlp type: {token_mlp}")
        self.mlp_proj = nn.Linear(mdssa_dim, dim)

    def forward(self, x, enc_out=None):
        B, L, C = x.shape
        H, W = int(math.sqrt(L)), int(math.sqrt(L))
        if enc_out is not None:
            x = torch.cat([x, enc_out], dim=2)
            
        shortcut = x

        x = self.norm1(x)
        x = self.mdassa(x, mask=None)
        x = rearrange(x, 'b h w c -> b (h w) c')
        y = x + shortcut
        
        x = x + shortcut
        x = self.norm2(x)
        x = self.mlp(x)
        x = y + self.drop_path(x)
       # x = y + x
        x = self.mlp_proj(x)
        return x
        
class MyModel(nn.Module):
    def __init__(self, img_size=256,dd_in=3, embed_dim=32, dropout_rate=0., drop_path_rate=0.1, use_dwt="Fourier"):
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim 

        self.num_enc_layers = 4

        self.input_proj = InputProjection(in_channels=dd_in,out_channels=embed_dim)
        self.pos_drop = nn.Dropout(p=dropout_rate)

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_enc_layers)]


        self.encoder_0 = EncoderBlock(dim=embed_dim, input_resolution=(self.img_size, self.img_size),
                                     num_heads=4, mlp_ratio=4, token_mlp="leff", drop_path=enc_dpr[0], norm_layer=nn.LayerNorm,
                                     act_layer=nn.GELU, drop=dropout_rate, freq_mlp="leff",use_dwt=use_dwt)
        
        self.downsample_0 = Downsample(embed_dim, embed_dim*2)
        self.encoder_1 = EncoderBlock(dim=embed_dim*2, input_resolution=(self.img_size//2, self.img_size//2), 
                                     num_heads=4, mlp_ratio=4, token_mlp="leff", drop_path=enc_dpr[1], norm_layer=nn.LayerNorm,
                                     act_layer=nn.GELU, drop=dropout_rate, freq_mlp="leff", use_dwt=use_dwt)
        self.downsample_1 = Downsample(embed_dim*2, embed_dim*4)
        self.encoder_2 = EncoderBlock(dim=embed_dim*4, input_resolution=(self.img_size//4, self.img_size//4),
                                     num_heads=4, mlp_ratio=4, token_mlp="leff", drop_path=enc_dpr[2], norm_layer=nn.LayerNorm,
                                     act_layer=nn.GELU, drop=dropout_rate, freq_mlp="leff", use_dwt=use_dwt)
        self.downsample_2 = Downsample(embed_dim*4, embed_dim*8)
        self.encoder_3 = EncoderBlock(dim=embed_dim*8, input_resolution=(self.img_size//8, self.img_size//8),
                                     num_heads=4, mlp_ratio=4, token_mlp="leff", drop_path=enc_dpr[3], norm_layer=nn.LayerNorm,
                                     act_layer=nn.GELU, drop=dropout_rate, freq_mlp="leff", use_dwt=use_dwt)
        self.downsample_3 = Downsample(embed_dim*8, embed_dim*16)

        self.bottleneck = DecoderBlock(dim=embed_dim*16, input_resolution=(self.img_size//16, self.img_size//16),
                                        num_heads=4, win_size=8, shift_size=0,
                                        mlp_ratio=4, token_mlp="leff", drop_path=0.0, norm_layer=nn.LayerNorm,
                                        act_layer=nn.GELU, drop=dropout_rate, token_projection="linear", enc_out=False, use_dwt=use_dwt)

        self.upsample_3 = Upsample(embed_dim*16, embed_dim*8)
        self.decoder_3 = DecoderBlock(dim=embed_dim*8, input_resolution=(self.img_size//8, self.img_size//8),
                                        num_heads=4, win_size=8, shift_size=0,
                                        mlp_ratio=4, token_mlp="leff", drop_path=0.0, norm_layer=nn.LayerNorm,
                                        act_layer=nn.GELU, drop=dropout_rate, token_projection="linear", enc_out=True, freq_attn_win_ratio=2, use_dwt=use_dwt)
        self.upsample_2 = Upsample(embed_dim*8, embed_dim*4)
        self.decoder_2 = DecoderBlock(dim=embed_dim*4, input_resolution=(self.img_size//4, self.img_size//4),
                                        num_heads=4, win_size=8, shift_size=0,
                                        mlp_ratio=4, token_mlp="leff", drop_path=0.0, norm_layer=nn.LayerNorm,
                                        act_layer=nn.GELU, drop=dropout_rate, token_projection="linear", enc_out=True, freq_attn_win_ratio=4, use_dwt=use_dwt)
        self.upsample_1 = Upsample(embed_dim*4, embed_dim*2)
        self.decoder_1 = DecoderBlock(dim=embed_dim*2, input_resolution=(self.img_size//2, self.img_size//2),
                                        num_heads=4, win_size=8, shift_size=0,
                                        mlp_ratio=4, token_mlp="leff", drop_path=0.0, norm_layer=nn.LayerNorm,
                                        act_layer=nn.GELU, drop=dropout_rate, token_projection="linear", enc_out=True, freq_attn_win_ratio=8, use_dwt=use_dwt)
        self.upsample_0 = Upsample(embed_dim*2, embed_dim)
        self.decoder_0 = DecoderBlock(dim=embed_dim, input_resolution=(self.img_size, self.img_size),
                                        num_heads=4, win_size=8, shift_size=0,
                                        mlp_ratio=4, token_mlp="leff", drop_path=0.0, norm_layer=nn.LayerNorm,
                                        act_layer=nn.GELU, drop=dropout_rate, token_projection="linear", enc_out=True, freq_attn_win_ratio=16, use_dwt=use_dwt) 
        
        self.output_proj = OutputProjection(in_channels=embed_dim, out_channel=dd_in, kernel_size=3, stride=1, norm_layer=None, act_layer=None)
        self.adaptive_pool_1 = nn.AdaptiveAvgPool2d(256*256*3)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        
        # x = self.adaptive_pool_1(x)
        # Input - x: (B, C, H, W) - B, 3, 256, 256
        y = self.input_proj(x)
        # print(f"Input Projection shape: {y.shape}")
        # y - y: (B, L, C) - B, 256*256, 32

        conv0 = self.encoder_0(y, mask=mask) # conv0 - (B, L, C) - B, 256*256, 32
        pool0 = self.downsample_0(conv0) # pool0 - (B, L/4, C*2) - B, 128*128, 64
        conv1 = self.encoder_1(pool0, mask=mask) # conv1 - (B, L/4, C*2) - B, 128*128, 64
        pool1 = self.downsample_1(conv1) # pool1 - (B, L/8, C*4) - B, 64*64, 128
        conv2 = self.encoder_2(pool1, mask=mask) # conv2 - (B, L/8, C*4) - B, 64*64, 128
        pool2 = self.downsample_2(conv2) # pool2 - (B, L/16, C*8) - B, 32*32, 256
        conv3 = self.encoder_3(pool2, mask=mask) # conv3 - (B, L/16, C*8) - B, 32*32, 256
        pool3 = self.downsample_3(conv3) # pool3 - (B, L/32, C*16) - B, 16*16, 512

        bottleneck = self.bottleneck(pool3)
        #print(f"Dimensions after Bottleneck {bottleneck.shape}")
        print("bottleneck", bottleneck.shape)
        up3 = self.upsample_3(bottleneck)
        #print(f"Dimensions after upsample: {up3.shape}")
        dec3 = self.decoder_3(up3, enc_out=conv3)
        #print(f"Dimensions after decoder: {dec3.shape}")
        up2 = self.upsample_2(dec3)
        #print(f"Dimensions after upsample2: {up2.shape}")
        dec2 = self.decoder_2(up2, enc_out=conv2)
        #print(f"Dimensions after decoder2: {dec2.shape}")
        up1 = self.upsample_1(dec2)
        #print(f"Dimensions after upsample1: {up1.shape}")
        dec1 = self.decoder_1(up1, enc_out=conv1)
        #print(f"Dimensions after decoder1: {dec1.shape}")
        up0 = self.upsample_0(dec1)
        #print(f"Dimensions after upsample0: {up0.shape}")
        dec0 = self.decoder_0(up0, enc_out=conv0)

        out = self.output_proj(dec0)
        out = out + x
        return out


class MyBigModel(nn.Module):
    def __init__(self, img_size=512, dd_in=3, embed_dim=32, dropout_rate=0., drop_path_rate=0.1, use_dwt="Fourier"):
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim

        self.num_enc_layers = 4

        self.input_proj = InputProjection(in_channels=dd_in, out_channels=embed_dim)
        self.pos_drop = nn.Dropout(p=dropout_rate)

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_enc_layers)]

        self.encoder_0 = EncoderBlock(dim=embed_dim, input_resolution=(self.img_size, self.img_size),
                                      num_heads=4, mlp_ratio=4, token_mlp="leff", drop_path=enc_dpr[0],
                                      norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, freq_mlp="leff", use_dwt=use_dwt)
        self.encoder_0_1 = EncoderBlock(dim=embed_dim, input_resolution=(self.img_size, self.img_size),num_heads=4, mlp_ratio=4, token_mlp="leff", drop_path=enc_dpr[0],
                                      norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, freq_mlp="leff", use_dwt=use_dwt)
        self.downsample_0 = Downsample(embed_dim, embed_dim * 2)
        self.encoder_1 = EncoderBlock(dim=embed_dim * 2, input_resolution=(self.img_size // 2, self.img_size // 2),
                                      num_heads=4, mlp_ratio=4, token_mlp="leff", drop_path=enc_dpr[1],
                                      norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, freq_mlp="leff", use_dwt=use_dwt)
        self.encoder_1_1 = EncoderBlock(dim=embed_dim * 2, input_resolution=(self.img_size//2, self.img_size//2),
                                        num_heads=4, mlp_ratio=4, token_mlp="leff", drop_path=enc_dpr[0],
                                        norm_layer=nn.LayerNorm,
                                        act_layer=nn.GELU, drop=dropout_rate, freq_mlp="leff", use_dwt=use_dwt
                                        )
        self.downsample_1 = Downsample(embed_dim * 2, embed_dim * 4)
        self.encoder_2 = EncoderBlock(dim=embed_dim * 4, input_resolution=(self.img_size // 4, self.img_size // 4),
                                      num_heads=4, mlp_ratio=4, token_mlp="leff", drop_path=enc_dpr[2],
                                      norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, freq_mlp="leff", use_dwt=use_dwt)
        self.encoder_2_1 = EncoderBlock(dim=embed_dim * 4, input_resolution=(self.img_size //4, self.img_size//4),
                                        num_heads=4, mlp_ratio=4, token_mlp="leff", drop_path=enc_dpr[0],
                                      norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, freq_mlp="leff", use_dwt=use_dwt)
        self.downsample_2 = Downsample(embed_dim * 4, embed_dim * 8)
        self.encoder_3 = EncoderBlock(dim=embed_dim * 8, input_resolution=(self.img_size // 8, self.img_size // 8),
                                      num_heads=4, mlp_ratio=4, token_mlp="leff", drop_path=enc_dpr[3],
                                      norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, freq_mlp="leff", use_dwt=use_dwt)
        self.encoder_3_1 = EncoderBlock(dim=embed_dim * 8, input_resolution=(self.img_size // 8, self.img_size // 8),num_heads=4, mlp_ratio=4, token_mlp="leff", drop_path=enc_dpr[0],
                                      norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, freq_mlp="leff", use_dwt=use_dwt)
        self.downsample_3 = Downsample(embed_dim * 8, embed_dim * 16)

        self.bottleneck = DecoderBlock(dim=embed_dim * 16, input_resolution=(self.img_size // 16, self.img_size // 16),
                                       num_heads=4, win_size=8, shift_size=0,
                                       mlp_ratio=4, token_mlp="leff", drop_path=0.0, norm_layer=nn.LayerNorm,
                                       act_layer=nn.GELU, drop=dropout_rate, token_projection="linear", enc_out=False,
                                       use_dwt=use_dwt)

        self.upsample_3 = Upsample(embed_dim * 16, embed_dim * 8)
        self.decoder_3 = DecoderBlock(dim=embed_dim * 8, input_resolution=(self.img_size // 8, self.img_size // 8),
                                      num_heads=4, win_size=8, shift_size=0,
                                      mlp_ratio=4, token_mlp="leff", drop_path=0.0, norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, token_projection="linear", enc_out=True,
                                      freq_attn_win_ratio=2, use_dwt=use_dwt)
        self.decoder_3_1 = DecoderBlock(dim=embed_dim * 8, input_resolution=(self.img_size // 8, self.img_size // 8),num_heads=4, win_size=8, shift_size=0,
                                      mlp_ratio=4, token_mlp="leff", drop_path=0.0, norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, token_projection="linear", enc_out=False,
                                      freq_attn_win_ratio=2, use_dwt=use_dwt)
        self.upsample_2 = Upsample(embed_dim * 8, embed_dim * 4)
        self.decoder_2 = DecoderBlock(dim=embed_dim * 4, input_resolution=(self.img_size // 4, self.img_size // 4),
                                      num_heads=4, win_size=8, shift_size=0,
                                      mlp_ratio=4, token_mlp="leff", drop_path=0.0, norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, token_projection="linear", enc_out=True,
                                      freq_attn_win_ratio=4, use_dwt=use_dwt)
        self.decoder_2_1 = DecoderBlock(dim=embed_dim * 4, input_resolution=(self.img_size // 4, self.img_size // 4),num_heads=4, win_size=8, shift_size=0,
                                      mlp_ratio=4, token_mlp="leff", drop_path=0.0, norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, token_projection="linear", enc_out=False,
                                      freq_attn_win_ratio=2, use_dwt=use_dwt)
        self.upsample_1 = Upsample(embed_dim * 4, embed_dim * 2)
        self.decoder_1 = DecoderBlock(dim=embed_dim * 2, input_resolution=(self.img_size // 2, self.img_size // 2),
                                      num_heads=4, win_size=8, shift_size=0,
                                      mlp_ratio=4, token_mlp="leff", drop_path=0.0, norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, token_projection="linear", enc_out=True,
                                      freq_attn_win_ratio=8, use_dwt=use_dwt)
        self.decoder_1_1 = DecoderBlock(dim=embed_dim * 2, input_resolution=(self.img_size // 2, self.img_size // 2),num_heads=4, win_size=8, shift_size=0,
                                      mlp_ratio=4, token_mlp="leff", drop_path=0.0, norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, token_projection="linear", enc_out=False,
                                      freq_attn_win_ratio=2, use_dwt=use_dwt)
        self.upsample_0 = Upsample(embed_dim * 2, embed_dim)
        self.decoder_0 = DecoderBlock(dim=embed_dim, input_resolution=(self.img_size, self.img_size),
                                      num_heads=4, win_size=8, shift_size=0,
                                      mlp_ratio=4, token_mlp="leff", drop_path=0.0, norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, token_projection="linear", enc_out=True,
                                      freq_attn_win_ratio=16, use_dwt=use_dwt)
        self.decoder_0_1 = DecoderBlock(dim=embed_dim, input_resolution=(self.img_size, self.img_size),num_heads=4, win_size=8, shift_size=0,
                                      mlp_ratio=4, token_mlp="leff", drop_path=0.0, norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, token_projection="linear", enc_out=False,
                                      freq_attn_win_ratio=2, use_dwt=use_dwt)

        self.output_proj = OutputProjection(in_channels=embed_dim, out_channel=dd_in, kernel_size=3, stride=1,
                                            norm_layer=None, act_layer=None)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        # Input - x: (B, C, H, W) - B, 3, 256, 256
        B,C,H,W = x.shape
        # print(x.shape)
        out_skip = x
        x = self.adaptive_pool(x)
        y = self.input_proj(x)

        # y - y: (B, L, C) - B, 256*256, 32

        conv0 = self.encoder_0(y, mask=mask)  # conv0 - (B, L, C) - B, 256*256, 32
        conv0 = self.encoder_0_1(conv0, mask=mask)
        pool0 = self.downsample_0(conv0)  # pool0 - (B, L/4, C*2) - B, 128*128, 64
        conv1 = self.encoder_1(pool0, mask=mask)  # conv1 - (B, L/4, C*2) - B, 128*128, 64
        conv1 = self.encoder_1_1(conv1, mask=mask)
        pool1 = self.downsample_1(conv1)  # pool1 - (B, L/8, C*4) - B, 64*64, 128
        conv2 = self.encoder_2(pool1, mask=mask)  # conv2 - (B, L/8, C*4) - B, 64*64, 128
        conv2 = self.encoder_2_1(conv2, mask=mask)
        pool2 = self.downsample_2(conv2)  # pool2 - (B, L/16, C*8) - B, 32*32, 256
        conv3 = self.encoder_3(pool2, mask=mask)  # conv3 - (B, L/16, C*8) - B, 32*32, 256
        conv3 = self.encoder_3_1(conv3, mask=mask)
        pool3 = self.downsample_3(conv3)  # pool3 - (B, L/32, C*16) - B, 16*16, 512

        bottleneck = self.bottleneck(pool3)
        # print(f"Dimensions after Bottleneck {bottleneck.shape}")
        up3 = self.upsample_3(bottleneck)
        # print(f"Dimensions after upsample: {up3.shape}")
        dec3 = self.decoder_3(up3, enc_out=conv3)
        dec3 = self.decoder_3_1(dec3)
        # print(f"Dimensions after decoder: {dec3.shape}")
        up2 = self.upsample_2(dec3)
        # print(f"Dimensions after upsample2: {up2.shape}")
        dec2 = self.decoder_2(up2, enc_out=conv2)
        dec2 = self.decoder_2_1(dec2)
        # print(f"Dimensions after decoder2: {dec2.shape}")
        up1 = self.upsample_1(dec2)
        # print(f"Dimensions after upsample1: {up1.shape}")
        dec1 = self.decoder_1(up1, enc_out=conv1)
        dec1 = self.decoder_1_1(dec1)

        up0 = self.upsample_0(dec1)

        dec0 = self.decoder_0(up0, enc_out=conv0)
        dec0 = self.decoder_0_1(dec0)
        b, l, c = dec0.shape
        h = w = int(math.sqrt(l))
        dec0 = dec0.transpose(1, 2).reshape(b, c, h, w).contiguous()
        # yh = math.sqrt(y.shape[1])
        # yw = math.sqrt(y.shape[1])
        # y = y.view(y.shape[0], y.shape[2], int(yh), int(yw))
        # dec0 = dec0 + y
        # dec0 = torch.nn.ConstantPad2d((W-dec0.shape[-1],0,H-dec0.shape[-2],0),0)(dec0)
        # dec0 = torchvision.transforms.functional.resize((dec0),(H,W))
        out = self.output_proj(dec0)
        
        out = out + x
        out = torchvision.transforms.functional.resize((out),(H,W))
        # print(out.shape)
        out_c1 = self.conv_super_enc(out)
        # print(out_c1.shape)
        out_c2 = self.conv_super_enc1(out_c1)
        # print(out_c2.shape)
        out_c3 = self.conv_super_enc2(out_c2)
        # print(out_c3.shape)
        # dec_c3 = self.conv_super_dec3(out_c3)
        dec_c2 = self.conv_super_dec2(out_c3)
        # print(dec_c2.shape)
        dec_c1 = self.conv_super_dec1(dec_c2+out_c2)#
        # print(dec_c1.shape)
        y = self.conv_super_dec(dec_c1 + out_c1)
        # print(out.shape)
        out = out + y
        return out

class MyBigFRFNModel(nn.Module):
    def __init__(self, img_size=512, dd_in=3, embed_dim=32, dropout_rate=0., drop_path_rate=0.1, use_dwt="Fourier"):
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim

        self.num_enc_layers = 4

        self.input_proj = InputProjection(in_channels=dd_in, out_channels=embed_dim)
        self.pos_drop = nn.Dropout(p=dropout_rate)

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_enc_layers)]

        self.encoder_0 = EncoderBlock(dim=embed_dim, input_resolution=(self.img_size, self.img_size),
                                      num_heads=4, mlp_ratio=4, token_mlp="frfn", drop_path=enc_dpr[0],
                                      norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, freq_mlp="frfn", use_dwt=use_dwt)
        self.encoder_0_1 = EncoderBlock(dim=embed_dim, input_resolution=(self.img_size, self.img_size), num_heads=4,
                                        mlp_ratio=4, token_mlp="frfn", drop_path=enc_dpr[0],
                                        norm_layer=nn.LayerNorm,
                                        act_layer=nn.GELU, drop=dropout_rate, freq_mlp="frfn", use_dwt=use_dwt)
        self.downsample_0 = Downsample(embed_dim, embed_dim * 2)
        self.encoder_1 = EncoderBlock(dim=embed_dim * 2, input_resolution=(self.img_size // 2, self.img_size // 2),
                                      num_heads=4, mlp_ratio=4, token_mlp="frfn", drop_path=enc_dpr[1],
                                      norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, freq_mlp="frfn", use_dwt=use_dwt)
        self.encoder_1_1 = EncoderBlock(dim=embed_dim * 2,
                                        input_resolution=(self.img_size // 2, self.img_size // 2),
                                        num_heads=4, mlp_ratio=4, token_mlp="frfn", drop_path=enc_dpr[0],
                                        norm_layer=nn.LayerNorm,
                                        act_layer=nn.GELU, drop=dropout_rate, freq_mlp="frfn", use_dwt=use_dwt
                                        )
        self.downsample_1 = Downsample(embed_dim * 2, embed_dim * 4)
        self.encoder_2 = EncoderBlock(dim=embed_dim * 4, input_resolution=(self.img_size // 4, self.img_size // 4),
                                      num_heads=4, mlp_ratio=4, token_mlp="frfn", drop_path=enc_dpr[2],
                                      norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, freq_mlp="frfn", use_dwt=use_dwt)
        self.encoder_2_1 = EncoderBlock(dim=embed_dim * 4,
                                        input_resolution=(self.img_size // 4, self.img_size // 4),
                                        num_heads=4, mlp_ratio=4, token_mlp="frfn", drop_path=enc_dpr[0],
                                        norm_layer=nn.LayerNorm,
                                        act_layer=nn.GELU, drop=dropout_rate, freq_mlp="frfn", use_dwt=use_dwt)
        self.downsample_2 = Downsample(embed_dim * 4, embed_dim * 8)
        self.encoder_3 = EncoderBlock(dim=embed_dim * 8, input_resolution=(self.img_size // 8, self.img_size // 8),
                                      num_heads=4, mlp_ratio=4, token_mlp="frfn", drop_path=enc_dpr[3],
                                      norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, freq_mlp="frfn", use_dwt=use_dwt)
        self.encoder_3_1 = EncoderBlock(dim=embed_dim * 8,
                                        input_resolution=(self.img_size // 8, self.img_size // 8), num_heads=4,
                                        mlp_ratio=4, token_mlp="frfn", drop_path=enc_dpr[0],
                                        norm_layer=nn.LayerNorm,
                                        act_layer=nn.GELU, drop=dropout_rate, freq_mlp="frfn", use_dwt=use_dwt)
        self.downsample_3 = Downsample(embed_dim * 8, embed_dim * 16)

        self.bottleneck = DecoderBlock(dim=embed_dim * 16,
                                       input_resolution=(self.img_size // 16, self.img_size // 16),
                                       num_heads=4, win_size=8, shift_size=0,
                                       mlp_ratio=4, token_mlp="frfn", drop_path=0.0, norm_layer=nn.LayerNorm,
                                       act_layer=nn.GELU, drop=dropout_rate, token_projection="linear",
                                       enc_out=False,
                                       use_dwt=use_dwt)

        self.upsample_3 = Upsample(embed_dim * 16, embed_dim * 8)
        self.decoder_3 = DecoderBlock(dim=embed_dim * 8, input_resolution=(self.img_size // 8, self.img_size // 8),
                                      num_heads=4, win_size=8, shift_size=0,
                                      mlp_ratio=4, token_mlp="frfn", drop_path=0.0, norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, token_projection="linear", enc_out=True,
                                      freq_attn_win_ratio=2, use_dwt=use_dwt)
        self.decoder_3_1 = DecoderBlock(dim=embed_dim * 8,
                                        input_resolution=(self.img_size // 8, self.img_size // 8), num_heads=4,
                                        win_size=8, shift_size=0,
                                        mlp_ratio=4, token_mlp="frfn", drop_path=0.0, norm_layer=nn.LayerNorm,
                                        act_layer=nn.GELU, drop=dropout_rate, token_projection="linear",
                                        enc_out=False,
                                        freq_attn_win_ratio=2, use_dwt=use_dwt)
        self.upsample_2 = Upsample(embed_dim * 8, embed_dim * 4)
        self.decoder_2 = DecoderBlock(dim=embed_dim * 4, input_resolution=(self.img_size // 4, self.img_size // 4),
                                      num_heads=4, win_size=8, shift_size=0,
                                      mlp_ratio=4, token_mlp="frfn", drop_path=0.0, norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, token_projection="linear", enc_out=True,
                                      freq_attn_win_ratio=4, use_dwt=use_dwt)
        self.decoder_2_1 = DecoderBlock(dim=embed_dim * 4,
                                        input_resolution=(self.img_size // 4, self.img_size // 4), num_heads=4,
                                        win_size=8, shift_size=0,
                                        mlp_ratio=4, token_mlp="frfn", drop_path=0.0, norm_layer=nn.LayerNorm,
                                        act_layer=nn.GELU, drop=dropout_rate, token_projection="linear",
                                        enc_out=False,
                                        freq_attn_win_ratio=2, use_dwt=use_dwt)
        self.upsample_1 = Upsample(embed_dim * 4, embed_dim * 2)
        self.decoder_1 = DecoderBlock(dim=embed_dim * 2, input_resolution=(self.img_size // 2, self.img_size // 2),
                                      num_heads=4, win_size=8, shift_size=0,
                                      mlp_ratio=4, token_mlp="frfn", drop_path=0.0, norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, token_projection="linear", enc_out=True,
                                      freq_attn_win_ratio=8, use_dwt=use_dwt)
        self.decoder_1_1 = DecoderBlock(dim=embed_dim * 2,
                                        input_resolution=(self.img_size // 2, self.img_size // 2), num_heads=4,
                                        win_size=8, shift_size=0,
                                        mlp_ratio=4, token_mlp="frfn", drop_path=0.0, norm_layer=nn.LayerNorm,
                                        act_layer=nn.GELU, drop=dropout_rate, token_projection="linear",
                                        enc_out=False,
                                        freq_attn_win_ratio=2, use_dwt=use_dwt)
        self.upsample_0 = Upsample(embed_dim * 2, embed_dim)
        self.decoder_0 = DecoderBlock(dim=embed_dim, input_resolution=(self.img_size, self.img_size),
                                      num_heads=4, win_size=8, shift_size=0,
                                      mlp_ratio=4, token_mlp="frfn", drop_path=0.0, norm_layer=nn.LayerNorm,
                                      act_layer=nn.GELU, drop=dropout_rate, token_projection="linear", enc_out=True,
                                      freq_attn_win_ratio=16, use_dwt=use_dwt)
        self.decoder_0_1 = DecoderBlock(dim=embed_dim, input_resolution=(self.img_size, self.img_size), num_heads=4,
                                        win_size=8, shift_size=0,
                                        mlp_ratio=4, token_mlp="frfn", drop_path=0.0, norm_layer=nn.LayerNorm,
                                        act_layer=nn.GELU, drop=dropout_rate, token_projection="linear",
                                        enc_out=False,
                                        freq_attn_win_ratio=2, use_dwt=use_dwt)

        self.output_proj = OutputProjection(in_channels=embed_dim, out_channel=dd_in, kernel_size=3, stride=1,
                                            norm_layer=None, act_layer=None)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        # Input - x: (B, C, H, W) - B, 3, 256, 256
        y = self.input_proj(x)

        # y - y: (B, L, C) - B, 256*256, 32

        conv0 = self.encoder_0(y, mask=mask)  # conv0 - (B, L, C) - B, 256*256, 32
        conv0 = self.encoder_0_1(conv0, mask=mask)
        pool0 = self.downsample_0(conv0)  # pool0 - (B, L/4, C*2) - B, 128*128, 64
        conv1 = self.encoder_1(pool0, mask=mask)  # conv1 - (B, L/4, C*2) - B, 128*128, 64
        conv1 = self.encoder_1_1(conv1, mask=mask)
        pool1 = self.downsample_1(conv1)  # pool1 - (B, L/8, C*4) - B, 64*64, 128
        conv2 = self.encoder_2(pool1, mask=mask)  # conv2 - (B, L/8, C*4) - B, 64*64, 128
        conv2 = self.encoder_2_1(conv2, mask=mask)
        pool2 = self.downsample_2(conv2)  # pool2 - (B, L/16, C*8) - B, 32*32, 256
        conv3 = self.encoder_3(pool2, mask=mask)  # conv3 - (B, L/16, C*8) - B, 32*32, 256
        conv3 = self.encoder_3_1(conv3, mask=mask)
        pool3 = self.downsample_3(conv3)  # pool3 - (B, L/32, C*16) - B, 16*16, 512

        bottleneck = self.bottleneck(pool3)
        print(f"Dimensions after Bottleneck {bottleneck.shape}")
        up3 = self.upsample_3(bottleneck)
        # print(f"Dimensions after upsample: {up3.shape}")
        dec3 = self.decoder_3(up3, enc_out=conv3)
        dec3 = self.decoder_3_1(dec3)
        # print(f"Dimensions after decoder: {dec3.shape}")
        up2 = self.upsample_2(dec3)
        # print(f"Dimensions after upsample2: {up2.shape}")
        dec2 = self.decoder_2(up2, enc_out=conv2)
        dec2 = self.decoder_2_1(dec2)
        # print(f"Dimensions after decoder2: {dec2.shape}")
        up1 = self.upsample_1(dec2)
        # print(f"Dimensions after upsample1: {up1.shape}")
        dec1 = self.decoder_1(up1, enc_out=conv1)
        dec1 = self.decoder_1_1(dec1)

        up0 = self.upsample_0(dec1)

        dec0 = self.decoder_0(up0, enc_out=conv0)
        dec0 = self.decoder_0_1(dec0)

        out = self.output_proj(dec0)

        out = out + x
        return out




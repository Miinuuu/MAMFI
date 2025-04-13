from __future__ import annotations

# Standard library imports
import math
from math import ceil
from functools import partial
from collections import namedtuple
from typing import Callable

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import stack, cat, Tensor
from torch.nn import Module, ModuleList, Linear

# Third-party library imports
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange, Reduce


# flex attention
# https://pytorch.org/blog/flexattention/
flex_attention = None


try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass


LinearNoBias = partial(Linear, bias = False)
AttnIntermediates = namedtuple('AttnIntermediates', ('value_residual', 'cached_key_values'))



def depad_if_needed(x, size, window_size):
    h, w = size
    pad_h = math.ceil(h / window_size[0]) * window_size[0] - h
    pad_w = math.ceil(w / window_size[1]) * window_size[1] - w
    if pad_h > 0 or pad_w > 0:  # remove the center-padding on feature
        return x[:, pad_h // 2 : pad_h // 2 + h, pad_w // 2 : pad_w // 2 + w, :].contiguous()
    return x


def pad_if_needed(x, size, window_size):
    n, h, w, c = size
    pad_h = math.ceil(h / window_size[0]) * window_size[0] - h
    pad_w = math.ceil(w / window_size[1]) * window_size[1] - w
    if pad_h > 0 or pad_w > 0:  # center-pad the feature on H and W axes
        return nn.functional.pad(
            x,
            (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        )
    return x


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depths=2,act_layer=nn.PReLU):
        super().__init__()
        layers = []
        for i in range(depths):
            if i == 0:
                layers.append(nn.Conv2d(in_dim, out_dim, 3,1,1))
            else:
                layers.append(nn.Conv2d(out_dim, out_dim, 3,1,1))
            layers.extend([
                act_layer(out_dim),
            ])
        self.conv = nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_layer=nn.RMSNorm):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = norm_layer(embed_dim)
        self.r=Rearrange('b c h w -> b (h w) c',c=embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x=self.r(x)
        #x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.r1= Rearrange('b (h w) d -> b d h w',d=dim )
        self.r2= Rearrange('b d h w -> b (h w) d',d=dim )
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.r1(x)
        x = self.dwconv(x)
        x = self.r2(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features,bias=False)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features,bias=False)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



from einops import repeat, rearrange, pack, unpack, einsum

class LearnablePad2D(nn.Module):
    """
    (B,H,W,C) 텐서에 대해:
      - pad_left, pad_right, pad_top, pad_bottom만큼 
        '학습 가능한' 패딩(메모리)을 붙여서 
        (B, H+pad_top+pad_bottom, W+pad_left+pad_right, C) 형태로 만든다.
      - remove_padding 메서드로 역으로 상하좌우를 잘라낼 수 있다.
    """
    
    def __init__(self,
                 in_height: int,
                 in_width: int,
                 in_channels: int,
                 pad_left: int,
                 pad_right: int,
                 pad_top: int,
                 pad_bottom: int):
        super().__init__()
        
        self.in_height = in_height
        self.in_width = in_width
        self.in_channels = in_channels
        
        self.pad_left   = pad_left
        self.pad_right  = pad_right
        self.pad_top    = pad_top
        self.pad_bottom = pad_bottom
        
        # ==============
        # 1) 좌/우 메모리
        # ==============
        if pad_left > 0:
            # shape: (1, H, pad_left, C)
            self.memory_left = nn.Parameter(
                torch.randn(1, in_height, pad_left, in_channels).cuda()
            )*0.02
        else:
            self.memory_left = None
        
        if pad_right > 0:
            # shape: (1, H, pad_right, C)
            self.memory_right = nn.Parameter(
                torch.randn(1, in_height, pad_right, in_channels).cuda()
            ) *0.02
        else:
            self.memory_right = None
        
        # =================
        # 2) 위/아래 메모리
        # =================
        # 좌/우 패딩을 붙인 후의 width = in_width + pad_left + pad_right
        new_width = in_width + pad_left + pad_right
        
        if pad_top > 0:
            # shape: (1, pad_top, new_width, C)
            self.memory_top = nn.Parameter(
                torch.randn(1, pad_top, new_width, in_channels).cuda()
            )*0.02
        else:
            self.memory_top = None
        
        if pad_bottom > 0:
            # shape: (1, pad_bottom, new_width, C)
            self.memory_bottom = nn.Parameter(
                torch.randn(1, pad_bottom, new_width, in_channels).cuda()
            )*0.02
        else:
            self.memory_bottom = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (B, H, W, C)
        return: shape (B, H + pad_top + pad_bottom, W + pad_left + pad_right, C)
        """
        B, H, W, C = x.shape
        # 안전 체크
        assert H == self.in_height,  f"Input height {H} != {self.in_height}"
        assert W == self.in_width,   f"Input width {W}  != {self.in_width}"
        assert C == self.in_channels,f"Input channels {C} != {self.in_channels}"
        
        # 1) 좌/우 메모리 붙이기
        if self.pad_left and self.memory_left is not None:

            #memL = self.memory_left.expand(B, -1, -1, -1)  # (B, H, pad_left, C)
            memL = repeat(self.memory_left, '1 h p c -> b h p c', b=x.shape[0]) # (B, H, pad_left, C)
            
            x = torch.cat([memL, x], dim=2)  # 너비(dim=2)에 concat
        
        if self.pad_right and self.memory_right is not None:
            #memR = self.memory_right.expand(B, -1, -1, -1) # (B, H, pad_right, C)
            memR = repeat(self.memory_right, '1 h p c -> b h p c', b=x.shape[0]) 

            x = torch.cat([x, memR], dim=2)  
        
        # 2) 위/아래 메모리 붙이기
        #   now x: (B, H, W+pad_left+pad_right, C)
        #   붙인 후: (B, H+pad_top+pad_bottom, W+pad_left+pad_right, C)
        if self.pad_top and self.memory_top is not None:
            #memT = self.memory_top.expand(B, -1, -1, -1)
            memT = repeat(self.memory_top, '1 p w c -> b p w c', b=x.shape[0]) 

            x = torch.cat([memT, x], dim=1)  # 높이(dim=1)에 concat
        
        if self.pad_bottom and self.memory_bottom is not None:
            #memB = self.memory_bottom.expand(B, -1, -1, -1)
            memB = repeat(self.memory_bottom, '1 p w c -> b p w c', b=x.shape[0]) 
            x = torch.cat([x, memB], dim=1)
        
        return x
    
    def remove_padding(self, x: torch.Tensor) -> torch.Tensor:
        """
        패딩을 붙인 상태의 텐서 x:
          (B, H + pad_top + pad_bottom, W + pad_left + pad_right, C)
        에서,
          상단 pad_top, 하단 pad_bottom,
          좌측 pad_left, 우측 pad_right
        을 잘라내 (B, H, W, C)로 복원한다.
        """
        B, Hp, Wp, C = x.shape
        
        # 슬라이싱 범위 계산
        #  top부터 ~ Hp - bottom
        top_start = self.pad_top
        top_end   = Hp - self.pad_bottom if self.pad_bottom > 0 else Hp
        
        # left부터 ~ Wp - right
        left_start = self.pad_left
        left_end   = Wp - self.pad_right if self.pad_right > 0 else Wp
        
        # crop
        x_cropped = x[:, top_start:top_end, left_start:left_end, :]
        
        # x_cropped shape이 (B,H,W,C)와 일치하는지 확인
        expected_H = self.in_height
        expected_W = self.in_width
        assert x_cropped.shape[1] == expected_H, f"Removed pad height {x_cropped.shape[1]} != {expected_H}"
        assert x_cropped.shape[2] == expected_W, f"Removed pad width {x_cropped.shape[2]} != {expected_W}"
        
        return x_cropped



def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def round_up_multiple(seq, mult):
    return ceil(seq / mult) * mult

def round_down_multiple(seq, mult):
    return seq // mult * mult

def pack_with_inverse(t, pattern):
    packed, packed_shape = pack(t, pattern)

    def inverse(out, inv_pattern = None):
        return unpack(out, packed_shape, default(inv_pattern, pattern))

    return packed, inverse

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

# sampling related

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    if temperature > 0.:
        t = t / temperature + gumbel_noise(t)
    return t.argmax(dim = -1, keepdim = True)

# min_p
# https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

# feedforward and attention

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.silu(gate) * x

# def FeedForward(dim, mult = 4,scale=2/3):
#     dim_inner = int(dim * mult * scale)

#     return nn.Sequential(
#         nn.RMSNorm(dim),
#         nn.Linear(dim, dim_inner * 2),
#         GEGLU(),
#         nn.Linear(dim_inner, dim)
#     )



class FeedForward(nn.Module):
    def __init__(self, dim ,mult=4,scale=2/3,cond=False):
        super().__init__()
        self.dim_inner = int(dim * mult * scale)
        self.cond=cond
        self.norm=  nn.RMSNorm(dim)
        self.Linear =  nn.Linear(dim, self.dim_inner * 2)
        if self.cond:
            self.dwconv = nn.Conv2d(self.dim_inner*2, self.dim_inner*2, 3, 1, 1, bias=True, groups=dim)
        self.act=    GEGLU()
        self.Linear2 =  nn.Linear(self.dim_inner, dim )


    def forward(self,x):
        x= self.norm(x)
        x = self.Linear(x)
        if self.cond:
            x=rearrange(x, 'b (h w ) d -> b d h w', d = self.dim_inner*2) 
            x = self.dwconv(x)
            x = rearrange(x, 'b d h w -> b (h w) d',d=self.dim_inner*2 )

        x = self.act(x)
        x = self.Linear2(x)
        return x




def create_mac_block_mask(seq_len, window_size, persist_mem_len, sliding = False):

    def create_mac_mask(_, __, q_idx, kv_idx):
        is_persist_mem = kv_idx < persist_mem_len
        kv_without_mem = kv_idx - persist_mem_len
        causal_mask = q_idx >= kv_without_mem

        if not sliding:
            block_diagonal = (q_idx // window_size) == (kv_without_mem // window_size)
            causal_mask = causal_mask & block_diagonal
        else:
            sliding_mask = (q_idx - kv_without_mem) <= window_size
            causal_mask = causal_mask & sliding_mask

        return is_persist_mem | (~is_persist_mem & causal_mask)

    block_mask = create_block_mask(create_mac_mask, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len + persist_mem_len, _compile = True)
    return block_mask

##########################################################################################
def pad_and_segment_with_inverse(
    seq,
    segment_len,
    fold_into_batch = True,
    inverse_remove_pad = True
):
    batch, seq_len = seq.shape[:2]
    next_seq_len_mult = round_up_multiple(seq_len, segment_len)

    padding = next_seq_len_mult - seq_len
    needs_pad = padding > 0

    if needs_pad:
        seq = F.pad(seq, (0, 0, 0, padding))

    if fold_into_batch:
        seq = rearrange(seq, 'b (w n) d -> (b w) n d', n = segment_len)

    def inverse(out):

        if fold_into_batch:
            out = rearrange(out, '(b w) ... n d -> b ... (w n) d', b = batch)

        if needs_pad and inverse_remove_pad:
            out = out[..., :-padding, :]

        return out

    return seq, inverse


def view(tensor,new_shape):
    if tensor.is_contiguous():
        new_tensor = tensor.view(new_shape)
    else:
        new_tensor = tensor.contiguous().view(new_shape)
    return tensor



def pad_and_segment_with_inverse_2d(
    seq,
    segment_len,
    fold_into_batch=True,
    inverse_remove_pad=True,
    fold_seq=True
):
    """
    2D 입력 텐서를 segment_len 크기의 윈도우로 패딩 및 분할하고,
    나중에 원본 형태로 복원할 수 있는 inverse 함수를 반환합니다.
    
    반환:
      (segmented_seq, attn_mask), inverse
      - segmented_seq: (B * num_windows, window_area, C) 형태의 윈도우들
      - attn_mask: 패딩 영역을 구분하는 어텐션 마스크 (필요시, 없으면 None)
      - inverse: 분할 결과를 원래 크기로 복원하는 함수
    """
    batch, seq_len_H, seq_len_W, C = seq.shape
    window_size = (segment_len[0], segment_len[1])
    
    next_seq_len_mult_H = round_up_multiple(seq_len_H, segment_len[0])
    next_seq_len_mult_W = round_up_multiple(seq_len_W, segment_len[1])
    pad_h = next_seq_len_mult_H - seq_len_H
    pad_w = next_seq_len_mult_W - seq_len_W
    needs_pad = pad_h > 0 or pad_w > 0

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    def window_partition(x, window_size, squeeze_last=False,fold_seq=True):
        ws0, ws1 = window_size
        # x: (B, H, W, C) → (B, H//ws0, ws0, W//ws1, ws1, C) → (B*h*w, ws0*ws1, C)
        if fold_seq :
            windows = rearrange(x, "B (h ws0) (w ws1) C -> (B h w) (ws0 ws1) C", ws0=ws0, ws1=ws1)
        else:
            windows = rearrange(x, "B (h ws0) (w ws1) C -> (B h w) ws0 ws1 C", ws0=ws0, ws1=ws1)

        return windows.squeeze(-1) if squeeze_last and windows.shape[-1] == 1 else windows
    
    
    padded_H = seq_len_H + pad_h
    padded_W = seq_len_W + pad_w

    attn_mask = None
    if needs_pad:
        seq = nn.functional.pad(seq, (0, 0, pad_left, pad_right, pad_top, pad_bottom))

    if fold_into_batch:
        seq = window_partition(seq, window_size,fold_seq)
    else:
        if fold_seq:
            seq=seq.view(batch,-1,C)


    def inverse(out,w=(None,None),fold_seq=True):

        if fold_into_batch:
            #out = rearrange(out, '(b w) ... n d -> b ... (w n) d', b = batch)

            whn= padded_H //  segment_len[0]
            wwn= padded_W //  segment_len[1]
            w_h_size = w[0] or (segment_len[0]) # 12
            w_w_size = w[1] or (segment_len[1]) # 12
        
            out = out.view(
                batch, whn, wwn, w_h_size, w_w_size, -1)
            out = out.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, whn  * w_h_size,   wwn  *  w_w_size, -1)
            if needs_pad and inverse_remove_pad:
                #out=out.reshape(batch, padded_H,padded_W,-1 )
                out = out[:, pad_top: pad_top + seq_len_H, pad_left: pad_left + seq_len_W , : ].contiguous()
            if fold_seq:
                out = out.view(batch,-1 , C)


        else:
            


            out = out.view(batch,padded_H,padded_W,-1)
            if needs_pad and inverse_remove_pad:
                #out=out.reshape(batch, padded_H,padded_W,-1 )
                out = out[:, pad_top: pad_top + seq_len_H, pad_left: pad_left + seq_len_W , : ].contiguous()
            if fold_seq:
                out=out.view(batch,-1,C)


        
        return out

    return seq ,inverse 

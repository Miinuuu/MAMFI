from rotary_embedding_torch import RotaryEmbedding

# PyTorch imports
import torch
from torch import nn, stack, cat, Tensor
from torch.nn import Module

# Standard library imports
from functools import partial
from collections import namedtuple
from math import ceil
from typing import Callable

# Local module imports
from .attend import *
from .layers import *

# Third-party library imports (einops)
from einops import repeat, rearrange, pack, unpack, einsum
from einops.layers.torch import Rearrange

# Optional flex attention imports
flex_attention = None
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass


#######################################################################################################
class SegmentedAttention_CSWIN_bi(Module):
    def __init__(
        self,
        dim,
        segment_len,
        num_persist_mem_tokens = 0,
        num_longterm_mem_tokens = 0,
        dim_head = 64,
        heads = 8,
        cross=False,
        causal=False,
        sliding = False,
        sliding_row_mask = False,
        sliding_bidirectional = False,
        accept_value_residual = False,
        attend_kwargs: dict = dict(),
        use_flex_attn = False
    ):
        super().__init__()
        print('SegmentedAttention_CSWIN_bi')
        self.norm = nn.RMSNorm(dim)
        self.cross=cross

        dim_inner = dim_head * heads

        self.rotary_emb = RotaryEmbedding(dim_head)

        self.attend = Attend(causal = causal , **attend_kwargs)

        # self.to_qkv = LinearNoBias(dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, dim)

        if self.cross: 
            self.to_q = LinearNoBias(dim, dim_inner )
            self.to_kv = LinearNoBias(dim, dim_inner * 2) #cross attention
        else: 
            self.to_qkv = LinearNoBias(dim, dim_inner * 3)


        self.to_learned_v_mix = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if accept_value_residual else None

        self.segment_len = segment_len
        self.num_longterm_mem_tokens = num_longterm_mem_tokens

        total_segment_len = segment_len + num_longterm_mem_tokens
        self.total_segment_len = total_segment_len

        self.sliding = sliding # sliding window attn - doubt their non-sliding results being the best. local attention with overlapping windows is very strong

        # sliding_row_mask      - previous / next window must be in the same row (column for the vertical branch)
        # sliding_bidirectional - also attend to the next window, with |q - k| <= total_segment_len instead of q - k <= total_segment_len

        assert sliding or not (sliding_row_mask or sliding_bidirectional), '`sliding_row_mask` / `sliding_bidirectional` require `sliding = True`'
        assert not (use_flex_attn and (sliding_row_mask or sliding_bidirectional)), 'flex attention path does not support `sliding_row_mask` / `sliding_bidirectional`'

        self.sliding_row_mask = sliding_row_mask
        self.sliding_bidirectional = sliding_bidirectional
        self.window_offsets = (-1, 0, 1) if sliding_bidirectional else (-1, 0)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.task_memory = nn.Parameter(torch.zeros(2, heads, num_persist_mem_tokens, dim_head))

        # flex attn related

        assert not (use_flex_attn and not exists(flex_attention)), 'you need to be on the latest pytorch with a cuda device available'
        self.use_flex_attn = use_flex_attn

        self.segment_len = segment_len
        self.num_persist_mem_tokens = num_persist_mem_tokens

    def forward_inference(
        self,
        token,
        cache,
        value_residual = None,
        output_gating = None,
    ):
        batch = token.shape[0]

        # attention

        token = self.norm(token)


        if self.cross:
            nwb=token.shape[0]
            seq_reverse = torch.cat([token[nwb//2:], token[:nwb//2]])
            k,v= self.to_kv(seq_reverse).chunk(2, dim = -1)
            q= self.to_q(token)
        else:
            q,k,v= self.to_qkv(token).chunk(3, dim = -1)


        # q, k, v = self.to_qkv(token).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        # value residual

        orig_v = v

        if exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(token)
            v = v.lerp(value_residual, mix)

        # caching

        ck, cv = cache
        k = cat((ck, k), dim = -2)
        v = cat((cv, v), dim = -2)

        next_cache = (k, v)

        # relative positions

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # fold

        q, k, v = tuple(rearrange(t, 'b h n d -> b h n d') for t in (q, k, v))

        # take care of persistent memory key / values

        pmk, pmv = repeat(self.persistent_memory, 'kv ... -> kv b ...', b = k.shape[0])

        # persistent memory

        k = cat((pmk, k), dim = -2)
        v = cat((pmv, v), dim = -2)

        # attention

        out, _ = self.attend(q, k, v)

        out = self.merge_heads(out)

        out = self.to_out(out)

        if exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)

    def forward_flex(
        self,
        seq,
        value_residual = None,
        flex_attn_fn: Callable | None = None,
        output_gating = None,
        cache = None
    ):

        assert not (exists(value_residual) ^ exists(self.to_learned_v_mix))

        batch, seq_len = seq.shape[:2]

        # attention

        seq = self.norm(seq)

        # q, k, v = self.to_qkv(seq).chunk(3, dim = -1)
        if self.cross:
            nwb=seq.shape[0]
            seq_reverse = torch.cat([seq[nwb//2:], seq[:nwb//2]])
            k,v= self.to_kv(seq_reverse).chunk(2, dim = -1)
            q= self.to_q(seq)
        else:
            q,k,v= self.to_qkv(seq).chunk(3, dim = -1)



        q, k, v = map(self.split_heads, (q, k, v))

        # value residual

        orig_v = v

        if exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)

        # caching

        next_cache = (k, v)

        # take care of persistent memory key / values

        pmk, pmv = repeat(self.persistent_memory, 'kv h n d -> kv b h n d', b = batch)

        # relative positions

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # persistent memory

        k = cat((pmk, k), dim = -2)
        v = cat((pmv, v), dim = -2)

        # prep flex attention

        if not exists(flex_attn_fn):
            block_mask = create_mac_block_mask(seq_len, self.total_segment_len, self.num_persist_mem_tokens, self.sliding)

            flex_attn_fn = partial(flex_attention, block_mask = block_mask)

        # attention

        out = flex_attn_fn(q, k, v)

        out = self.merge_heads(out)

        out = self.to_out(out)

        if exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)

    def forward(
        self,
        seq,
        B,
        H,
        W,
        idx,
        value_residual = None,
        flex_attn_fn: Callable | None = None,
        disable_flex_attn = False,
        output_gating = None, #retrieve
        cache = None,
        kv_valid = None # (H * W,) bool, False for zero-pad tokens
    ):
        is_inferencing = exists(cache)

        if is_inferencing:
            assert seq.shape[-2] == 1
            return self.forward_inference(seq, cache, value_residual, output_gating = output_gating)

        if seq.is_cuda and self.use_flex_attn and not disable_flex_attn:
            return self.forward_flex(seq, value_residual, flex_attn_fn, output_gating = output_gating, cache = cache)

        assert not (exists(value_residual) ^ exists(self.to_learned_v_mix))

        segment_len, num_longterm_mem_tokens = self.segment_len, self.num_longterm_mem_tokens
        total_segment_len = segment_len + num_longterm_mem_tokens

        batch, seq_len , dim = seq.shape
        seq=seq.view(B,H,W,-1)

        # auto pad to multiple
        seq, inverse_segment = pad_and_segment_with_inverse_2d(seq, (1,total_segment_len), fold_into_batch = False,inverse_remove_pad=True,fold_seq=True)

        if exists(kv_valid):
            kv_valid, _ = pad_and_segment_with_inverse_2d(kv_valid.view(1, H, W, 1).float(), (1, total_segment_len), fold_into_batch = False, inverse_remove_pad = True, fold_seq = True)
            kv_valid = kv_valid.view(-1) > 0.5
            assert kv_valid.numel() == seq.shape[-2]


        # attention seq= #B,N,C

        seq = self.norm(seq)

        if self.cross:
            nwb=seq.shape[0]
            seq_reverse = torch.cat([seq[nwb//2:], seq[:nwb//2]])
            k,v= self.to_kv(seq_reverse).chunk(2, dim = -1)
            q= self.to_q(seq)
        else:
            q,k,v= self.to_qkv(seq).chunk(3, dim = -1)

           
        q, k, v = map(self.split_heads, (q, k, v))

        # value residual

        orig_v = v  #B,h,N,d/h

        if exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)

        # relative positions

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # fold

        q, k, v = tuple(rearrange(t, 'b h (w n) d -> (b w) h n d', n = total_segment_len) for t in (q, k, v))

        # maybe sliding for cpu

        attend_kwargs = dict()

        num_windows = seq.shape[-2] // total_segment_len
        offsets = self.window_offsets if self.sliding else (0,)

        if self.sliding:
            # concat neighboring windows in the order of `offsets`, which the mask columns follow
            k, v = tuple(rearrange(t, '(b w) ... -> b w ...', b = batch) for t in (k, v))
            k, v = tuple(pad_at_dim(t, (-offsets[0], offsets[-1]), value = 0., dim = 1) for t in (k, v))
            k, v = tuple(cat([t[:, (o - offsets[0]):(o - offsets[0] + num_windows)] for o in offsets], dim = -2) for t in (k, v))
            k, v = tuple(rearrange(t, 'b w ... -> (b w) ...') for t in (k, v))

        # take care of masking

        if self.sliding or exists(kv_valid):
            window_mask = create_window_attn_mask(
                num_windows,
                (seq.shape[-2] // H) // total_segment_len,
                total_segment_len,
                offsets = offsets,
                bidirectional = self.sliding_bidirectional,
                respect_rows = self.sliding_row_mask,
                kv_valid = kv_valid,
                device = seq.device
            )

            window_mask = F.pad(window_mask, (self.num_persist_mem_tokens, 0), value = True)

            window_mask = repeat(window_mask, 'w i j -> (b w) 1 i j', b = batch)
            attend_kwargs.update(mask = window_mask)

        # take care of persistent memory key / values

        tmk, tmv = repeat(self.task_memory, 'kv ... -> kv b ...', b = k.shape[0])

        # persistent memory

        k = cat((tmk, k), dim = -2)
        v = cat((tmv, v), dim = -2)

        # attention

        out, _ = self.attend(q, k, v, **attend_kwargs)

        out = self.merge_heads(out)

        out = self.to_out(out)

        out = rearrange(out, '(b w) n d -> b (w n) d', b = batch)

        out = inverse_segment(out)

        return out, orig_v
#######################################################################################################
#######################################################################################################

import math
import tqdm
from copy import deepcopy
from functools import partial
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import stack, cat
from torch.nn import Module, ModuleList, Linear

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange, Reduce

# 로컬 모듈 / 패키지
from .neural_memory import *
from .attention import *
from .layers import *
from .continuous_axial_positional_embedding import ContinuousAxialPositionalEmbedding


class MAMFE_Block(nn.Module):
    def __init__(self,
                 dim, 
                 num_heads, 
                 window_size=8, 
                 shift_size=0, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.RMSNorm,
                 
                depth=2,
                segment_len=64,
                neural_memory_segment_len = None,
                neural_mem_gate_attn_output = False,
                neural_memory_add_value_residual = False,
                num_longterm_mem_tokens = 4,
                num_persist_mem_tokens = 0,
                neural_memory_batch_size = None,
                neural_memory_qkv_receives_diff_views = False,
                dim_head = 32,
                heads = 4,
                ff_mult = 4,
                num_residual_streams = 4,
                neural_memory_model: Module | None = None,
                neural_memory_kwargs: dict = dict(),
                neural_memory_layers: tuple[int, ...] | None = None,
                
                use_flex_attn = False,
                sliding_window_attn = True,
                neural_mem_weight_residual = True,#true
                is_first_neural_mem=False,      
                idx=False  
                 ):
        

        super().__init__()
        print('TitanBlock_CSWIN_PAD_8_bi')
        self.window_size = window_size
        if not isinstance(self.window_size, (tuple, list)):
            self.window_size = to_2tuple(window_size)
        self.shift_size = shift_size
        if not isinstance(self.shift_size, (tuple, list)):
            self.shift_size = to_2tuple(shift_size)

        # self.norm1 = norm_layer(dim)
        self.dim=dim

        self.segment_len = segment_len # 8 # window 길이  
        self.num_longterm_mem_tokens = num_longterm_mem_tokens # 메모리 길이 4
        self.longterm_memsh0 = nn.Parameter(torch.randn(num_longterm_mem_tokens, dim//2) * 0.02)
        self.longterm_memsh1 = nn.Parameter(torch.randn(num_longterm_mem_tokens, dim//2) * 0.02)
        self.longterm_memsv0 = nn.Parameter(torch.randn(num_longterm_mem_tokens, dim//2) * 0.02)
        self.longterm_memsv1 = nn.Parameter(torch.randn(num_longterm_mem_tokens, dim//2) * 0.02)

        #고민 중이다.  그냥 옆에다 붙이는게 맞을지 

        self.sliding_window_attn = sliding_window_attn
        self.attn_window_size = segment_len + num_longterm_mem_tokens

        # init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, dim = dim, add_stream_embed = True, disable = num_residual_streams == 1)

        self.layers = ModuleList([])        
        self.neural_memory_segment_len = default(neural_memory_segment_len, num_longterm_mem_tokens + segment_len)

        self.axial_pos_emb0 = ContinuousAxialPositionalEmbedding(dim = dim//2, num_axial_dims = 2) # 2d = 2 
        self.axial_pos_emb1 = ContinuousAxialPositionalEmbedding(dim = dim//2, num_axial_dims = 2) # 2d = 2 

        layers = tuple(range(1, 1 + 1)) # depth == 2  

        neural_memory_layers = default(neural_memory_layers, layers) #(2,4)
        #self.neural_mem_weight_re
        self.neural_mem_weight_residual = neural_mem_weight_residual # True
        self.is_first_neural_mem = is_first_neural_mem

        for layer in layers:
            is_first = layer == 1 if  idx==0 else False

            # attention and feedforward

            attn0= SegmentedAttention_CSWIN_bi(
                dim = dim//2,
                dim_head = dim_head,
                heads = heads//2,
                cross=True,
                causal=False,
                segment_len = segment_len, # 8,8 
                use_flex_attn = use_flex_attn, # false
                accept_value_residual = not is_first,
                num_longterm_mem_tokens =num_longterm_mem_tokens, # 4,4
                num_persist_mem_tokens = num_persist_mem_tokens,
                sliding = sliding_window_attn
            )
            attn1 = SegmentedAttention_CSWIN_bi(
                dim = dim//2,
                dim_head = dim_head,
                heads = heads//2,
                cross=True,
                causal=False,
                segment_len = segment_len, # 8,8 
                use_flex_attn = use_flex_attn, # false
                accept_value_residual = not is_first,
                num_longterm_mem_tokens =num_longterm_mem_tokens, # 4,4
                num_persist_mem_tokens = num_persist_mem_tokens,
                sliding = sliding_window_attn
            )

            mem = None

            if layer in neural_memory_layers:
                nm_dim=dim//2
                mem0 = NeuralMemory_bi(
                    dim =nm_dim,
                    chunk_size = self.neural_memory_segment_len,
                    batch_size = neural_memory_batch_size,
                    model = deepcopy(neural_memory_model),
                    model2 = deepcopy(neural_memory_model),
                    qkv_receives_diff_views = True,
                    accept_weight_residual = neural_mem_weight_residual and not self.is_first_neural_mem,
                    **neural_memory_kwargs
                )
                mem1 = NeuralMemory_bi(
                    dim = nm_dim,
                    chunk_size = self.neural_memory_segment_len,
                    batch_size = neural_memory_batch_size,
                    model = deepcopy(neural_memory_model),
                    model2 = deepcopy(neural_memory_model),
                    qkv_receives_diff_views = True,
                    accept_weight_residual = neural_mem_weight_residual and not self.is_first_neural_mem,
                    **neural_memory_kwargs
                )

                self.is_first_neural_mem = False

            ff = FeedForward(dim = dim, mult = ff_mult,scale=1,cond=False)

            self.layers.append(ModuleList([
                mem0,
                mem1,
                attn0,
                attn1,
                ff,
            ]))

        self.norm = nn.RMSNorm(dim)
        self.prj=nn.Linear(dim,dim)
        # whether to gate the attention output with the retrieved memories
        self.gate_attn_output = neural_mem_gate_attn_output
        # zero for maybe aux loss + device
        self.register_buffer('zero', torch.tensor(0.), persistent = False)
        # flex attn related
        assert not (use_flex_attn and not exists(flex_attention)), 'you need to be on the latest pytorch with a cuda device available'
        self.use_flex_attn = use_flex_attn
        self.num_persist_mem_tokens = num_persist_mem_tokens


    def initialize_persistent_memory(self, persistent_memory_dim: int):
        persistent_memory = nn.Parameter(torch.empty(1, 1, persistent_memory_dim, requires_grad=True))
        nn.init.xavier_normal_(persistent_memory)
        return persistent_memory


    def seq_index_is_longterm(
        self,
        seq_index
    ):
        total_segment_len, segment_len = self.attn_window_size, self.segment_len
        return ((seq_index % total_segment_len + 1) - segment_len) > 0

    def seq_len_with_longterm_mem(
        self,
        seq_len
    ):
        assert seq_len > 0

        segment_len, num_mem = self.segment_len, self.num_longterm_mem_tokens
        return ((seq_len - 1) // segment_len) * num_mem + seq_len

    def seq_len_with_longterm_mem2(
        self,
        seq_len
    ):
        assert seq_len > 0

        segment_len, num_mem = self.segment_len, self.num_longterm_mem_tokens

        
        return ((seq_len ) // segment_len) * num_mem + seq_len

    def seq_len_with_longterm_mem3(
        self,
        seq_len
    ):
        assert seq_len > 0

        segment_len, num_mem = self.segment_len, self.num_longterm_mem_tokens
        return  ceil(seq_len  /segment_len)  * (num_mem+segment_len)
        # 56//16 = 3 * 4 + 56 =  68 <-> 20-20-20-20 (8+8P+4m) -> 80
        # 32//16 = 2 *4 + 32 = 40  , 20 20 


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

    def forward(self
        ,x
        # value residual
        ,value_residual = (None,None)
        # neural mem weight residual
        ,mem_weight_residual = (None,None)
        ,B = None
        ,H = None
        ,W = None
        ,return_loss = False
        ,return_loss_breakdown = False
        ,disable_flex_attn = False
        ,cache = None
        ,return_cache = False
        ,factorized_pos_emb = None
        ):

        batch,seq_len,c =  x.shape #  2, 32* 32 , 128
        neural_mem_segment_len= self.neural_memory_segment_len
        segment_len=self.segment_len
        num_longterm_mem_tokens=self.num_longterm_mem_tokens
        attn_window_size = self.attn_window_size
        seq_H_len_with_mem = self.seq_len_with_longterm_mem3(H)
        seq_W_len_with_mem = self.seq_len_with_longterm_mem3(W)

        # intersperse longterm memory
        # 가로세로 패딩이 필요할 것 같다.
        x= x.view(B,H,W,self.dim)

        x0=x[...,:c//2]
        x1=x[...,c//2:]
        #  32,64,  32 * 4  128 개 
        x0, inverse_segment0 = pad_and_segment_with_inverse_2d(x0, (1,segment_len), fold_into_batch=True,inverse_remove_pad = False,fold_seq=True)
        x1, inverse_segment1 = pad_and_segment_with_inverse_2d(x1, (segment_len,1), fold_into_batch=True,inverse_remove_pad = False,fold_seq=True)

        memsh0 = repeat(self.longterm_memsh0, 'n d -> b n d', b = x0.shape[0]//2)
        memsh1 = repeat(self.longterm_memsh1, 'n d -> b n d', b = x0.shape[0]//2)
        memsh= torch.cat([memsh0,memsh1])
        memsv0 = repeat(self.longterm_memsv0, 'n d -> b n d', b = x1.shape[0]//2)
        memsv1 = repeat(self.longterm_memsv1, 'n d -> b n d', b = x1.shape[0]//2)
        memsv= torch.cat([memsv0,memsv1])


        x0, inverse_pack_mems0 = pack_with_inverse((x0, memsh), 'b * d')
        x1, inverse_pack_mems1 = pack_with_inverse((x1, memsv), 'b * d')
        #BW,N+P,D
        x0 = inverse_segment0(x0,w=(1,attn_window_size),fold_seq=False)
        x1 = inverse_segment1(x1,w=(attn_window_size,1),fold_seq=False)
        
        x0 = x0[:, :, :seq_W_len_with_mem]
        x1 = x1[:, :seq_H_len_with_mem , : ].transpose(1,2).contiguous()

        pos_emb0 = self.axial_pos_emb0.forward_with_seq_len(H*seq_W_len_with_mem, (neural_mem_segment_len,), factorized = factorized_pos_emb)
        pos_emb1 = self.axial_pos_emb1.forward_with_seq_len(seq_H_len_with_mem*W, (neural_mem_segment_len,), factorized = factorized_pos_emb)

        x0 = x0.view(B,-1,self.dim//2) + pos_emb0
        x1 = x1.view(B,-1,self.dim//2) + pos_emb1
        
        
        # To DO
        # 나중에 빠르게 할 때 다시 수정 
        use_flex_attn = x.is_cuda and self.use_flex_attn and not disable_flex_attn
        flex_attn0_fn = None 
        flex_attn1_fn = None 

        if use_flex_attn:
            block0_mask = create_mac_block_mask(H*seq_W_len_with_mem, self.attn_window_size, self.num_persist_mem_tokens, self.sliding_window_attn)
            block1_mask = create_mac_block_mask(W*seq_H_len_with_mem, self.attn_window_size, self.num_persist_mem_tokens, self.sliding_window_attn)
            flex_attn0_fn = partial(flex_attention, block_mask = block0_mask)
            flex_attn1_fn = partial(flex_attention, block_mask = block1_mask)


        # value residual
        value0_residual,value1_residual = value_residual

        # neural mem weight residual
        mem0_weight_residual,mem1_weight_residual= mem_weight_residual

        # layers for the neural mem to select the qkv inputs from

        for mem0,mem1 ,attn0,attn1, ff in self.layers:

            retrieved0 = None
            retrieved1 = None
            attn0_out_gates = None
            attn1_out_gates = None

            # maybe neural memory


            attn0_out, values0 = attn0(
                x0,
                B,
                H,
                seq_W_len_with_mem,
                idx=0,
                value_residual = value0_residual,
                disable_flex_attn = disable_flex_attn,
                flex_attn_fn = flex_attn0_fn,
                output_gating = attn0_out_gates,
            )

            attn1_out, values1 = attn1(
                x1,
                B,
                W,
                seq_H_len_with_mem,
                idx=1,
                value_residual = value1_residual,
                disable_flex_attn = disable_flex_attn,
                flex_attn_fn = flex_attn1_fn,
                output_gating = attn1_out_gates,
            )



            if exists(mem0) and exists(mem1) :
                qkv_mem_input0 = stack((attn0_out,attn0_out,attn0_out)) # attention 입력에서 norm을 하니까
                qkv_mem_input1 = stack((attn1_out,attn1_out,attn1_out)) # attention 입력에서 norm을 하니까
             

                retrieved0, next_neural_mem0_cache = mem0.forward(
                    qkv_mem_input0,
                    state =None,
                    prev_weights = mem0_weight_residual
                )
                retrieved1, next_neural_mem1_cache = mem1.forward(
                    qkv_mem_input1,
                    state = None,
                    prev_weights = mem1_weight_residual
                )

                if self.neural_mem_weight_residual:
                    mem0_weight_residual = next_neural_mem0_cache.updates

                if self.neural_mem_weight_residual:
                    mem1_weight_residual = next_neural_mem1_cache.updates

            #memory update
            
            if self.gate_attn_output : 
                attn0_out = attn0_out *  retrieved0.sigmoid()
                attn1_out = attn1_out * retrieved1.sigmoid()
            else:
                attn0_out = attn0_out +  retrieved0
                attn1_out = attn1_out +  retrieved1
 
            value0_residual = values0
            value1_residual = values1


            # Process attn0_out
            attn0_out = attn0_out.view(B, H, seq_W_len_with_mem, self.dim // 2)
            attn0_out, inverse_segment0 = pad_and_segment_with_inverse_2d(
                attn0_out, (1, attn_window_size), fold_into_batch=True, inverse_remove_pad=False)
            attn0_out, _ = inverse_pack_mems0(attn0_out)
            attn0_out = inverse_segment0(attn0_out, (1, segment_len),False)
            attn0_out= depad_if_needed(attn0_out, (H,W),(1,segment_len))
            

            # Process attn1_out
            attn1_out = attn1_out.view(B, W, seq_H_len_with_mem, self.dim // 2).transpose(1,2).contiguous()
            attn1_out, inverse_segment1 = pad_and_segment_with_inverse_2d(
                attn1_out, (attn_window_size, 1), fold_into_batch=True, inverse_remove_pad=False)
            attn1_out, _ = inverse_pack_mems1(attn1_out)
            attn1_out = inverse_segment1(attn1_out, (segment_len, 1),False)
            attn1_out= depad_if_needed(attn1_out, (H,W),(segment_len,1))

            # Concatenate along the feature dimension and project
            attend = self.prj(torch.cat([attn0_out, attn1_out], -1))

            x0 = x0.view(B, H, seq_W_len_with_mem, self.dim // 2)
            x0, inverse_segment0 = pad_and_segment_with_inverse_2d(
                x0, (1, attn_window_size), fold_into_batch=True, inverse_remove_pad=False)
            x0, _ = inverse_pack_mems0(x0)
            x0 = inverse_segment0(x0, (1, segment_len),False)
            x0= depad_if_needed(x0, (H,W),(1,segment_len))
            

            x1 = x1.view(B, W, seq_H_len_with_mem, self.dim // 2).transpose(1,2).contiguous()
            x1, inverse_segment1 = pad_and_segment_with_inverse_2d(
                x1, (attn_window_size, 1), fold_into_batch=True, inverse_remove_pad=False)
            x1, _ = inverse_pack_mems1(x1)
            x1 = inverse_segment1(x1, (segment_len, 1),False)
            x1= depad_if_needed(x1, (H,W),(segment_len,1))
    
    
            # Concatenate the processed tensors and add the attend residual
            x = (torch.cat([x0, x1], -1) + attend).view(B, -1, self.dim )

            # Apply feed-forward network and normalization
            x = self.norm(x + ff(x))
        return x, (value0_residual,value1_residual),(mem0_weight_residual,mem1_weight_residual)
    
    
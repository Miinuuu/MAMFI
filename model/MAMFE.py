import math

import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange, Reduce
from .memory_models import *
from .MAMFE_Blocks import *
from .layers import *

class MAMFE(nn.Module):
    def __init__(self, 
                 in_chans=3, 
                 embed_dims=[8, 16, 32, 64, 128], 
                 num_heads=[4, 8], 
                 mlp_ratios=[4, 4], 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0.,
                 attn_drop_rate=0., 
                 drop_path_rate=0., 
                 norm_layer=nn.RMSNorm,
                 depths=[2, 2, 2, 1, 1], 
                 window_sizes=[8, 8]
                
                ,NEURAL_MEMORY_DEPTH = 4 
                ,NUM_PERSIST_MEM = 12
                ,NUM_LONGTERM_MEM = 4
                ,NEURAL_MEM_LAYERS = (2,4)                            # layers 2, 4, 6 have neural memory, can add more
                ,NEURAL_MEM_GATE_ATTN_OUTPUT = False
                ,NEURAL_MEM_MOMENTUM = True
                ,NEURAL_MEM_MOMENTUM_ORDER = 1
                ,NEURAL_MEM_QK_NORM = True
                ,NEURAL_MEM_MAX_LR = 1e-1
                ,USE_MEM_ATTENTION_MODEL = False
                ,USE_MEM_SWIGLUMLP_MODEL = False
                ,USE_MEM_FACMLP_MODEL = False
                ,USE_MEM_GATEDRESMLP_MODEL = False
                ,USE_MEM_MLP_MODEL = False
                
                
                ,NEURAL_MEM_SEGMENT_LEN = 16                      # set smaller for more granularity for learning rate / momentum etc
                ,NEURAL_MEM_BATCH_SIZE = 32                     # set smaller to update the neural memory weights more often as it traverses the sequence
                ,SLIDING_WINDOWS = False
                ,STORE_ATTN_POOL_CHUNKS = True                   # whether to use attention pooling for chunk derived momentum, per-layer lr mod, decay
                ,MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
                ,NEURAL_MEM_WEIGHT_RESIDUAL = True               # learning to accept contributions from the weights of the previous neural mem layer brings about significant improvements. this was improvised and not in the paper, but inspired by the value residual learning free lunch paper
                ,NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True        # will allow the neural memory to select what layers from which to derive queries / keys / values, effectively allowing it to graft itself to the transformer in any way to be beneficial. this is to address an issue from a phd student who noted that the mem network is learning nothing more than wk @ wv. this also generalizes all possible ways to connect the neural memory to a transformer, a sort of NAS
                ,USE_ACCELERATED_SCAN = False
                ,USE_FLEX_ATTN = False
                
                , **kwarg):
        super().__init__()
        print('MAMFE')

        self.depths = depths
        self.num_stages = len(embed_dims)



        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(
                                                        patch_size = 3,
                                                        stride     = 2,
                                                        norm_layer = nn.LayerNorm,
                                                        in_chans   = embed_dims[i - 1],
                                                        embed_dim  = embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(
                                                        patch_size = 3,
                                                        stride     = 2,
                                                        norm_layer = nn.LayerNorm,
                                                        in_chans   = embed_dims[i - 1],
                                                        embed_dim  = embed_dims[i])

                    if USE_MEM_ATTENTION_MODEL:
                        neural_memory_model = MemoryAttention(
                            dim = embed_dims[i] // num_heads[i-self.conv_stages],
                            scale=None,
                            expansion_factor = 2.

                        )
                    elif USE_MEM_MLP_MODEL:
                        neural_memory_model = MemoryMLP(
                            dim = embed_dims[i] // num_heads[i-self.conv_stages],
                            depth = NEURAL_MEMORY_DEPTH,
                            expansion_factor = 2.,
                            act=F.gelu
                        )
                    elif USE_MEM_SWIGLUMLP_MODEL:
                        neural_memory_model = MemorySwiGluMLP(
                            dim = embed_dims[i] // num_heads[i-self.conv_stages],
                            depth = NEURAL_MEMORY_DEPTH,
                            expansion_factor=2
                        )

                    elif USE_MEM_FACMLP_MODEL :
                        neural_memory_model = FactorizedMemoryMLP(
                            dim = embed_dims[i] // num_heads[i-self.conv_stages],
                            depth = NEURAL_MEMORY_DEPTH,
                            k=32
                        )
                    elif USE_MEM_GATEDRESMLP_MODEL :
                        neural_memory_model = GatedResidualMemoryMLP(
                            dim = embed_dims[i] // num_heads[i-self.conv_stages],
                            depth = NEURAL_MEMORY_DEPTH,
                            expansion_factor = 4.
                        )

                    block = nn.ModuleList([MAMFE_Block(
                        dim=embed_dims[i], 
                        num_heads=num_heads[i-self.conv_stages], 
                        window_size=window_sizes[i-self.conv_stages], 
                        shift_size= 0 if (j % 2) == 0 else window_sizes[i-self.conv_stages] // 2,
                        mlp_ratio=mlp_ratios[i-self.conv_stages], 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=dpr[cur + j], 
                        norm_layer=norm_layer,

                        #depth = NEURAL_MEMORY_DEPTH,
                        segment_len = window_sizes[i-self.conv_stages],
                        num_persist_mem_tokens = NUM_PERSIST_MEM,
                        num_longterm_mem_tokens = NUM_LONGTERM_MEM,
                        neural_memory_layers = NEURAL_MEM_LAYERS,
                        neural_memory_segment_len = NEURAL_MEM_SEGMENT_LEN,
                        neural_memory_batch_size = NEURAL_MEM_BATCH_SIZE,
                        neural_mem_gate_attn_output = NEURAL_MEM_GATE_ATTN_OUTPUT,
                        neural_mem_weight_residual = NEURAL_MEM_WEIGHT_RESIDUAL,
                        neural_memory_qkv_receives_diff_views = NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
                        use_flex_attn = USE_FLEX_ATTN,
                        sliding_window_attn = SLIDING_WINDOWS,
                        neural_memory_model = neural_memory_model,
                        heads = num_heads[i-self.conv_stages],
                        dim_head = embed_dims[i]//num_heads[i-self.conv_stages],
                        is_first_neural_mem = j==0,
                        idx = j,

                        #Memory Config 
                        neural_memory_kwargs=dict(
                        dim_head = embed_dims[i]//num_heads[i-self.conv_stages],
                        heads = num_heads[i-self.conv_stages]//2,
                        attn_pool_chunks = STORE_ATTN_POOL_CHUNKS,
                        qk_rmsnorm = NEURAL_MEM_QK_NORM,
                        momentum = NEURAL_MEM_MOMENTUM,
                        momentum_order = NEURAL_MEM_MOMENTUM_ORDER,
                        default_step_transform_max_lr = NEURAL_MEM_MAX_LR,
                        use_accelerated_scan = USE_ACCELERATED_SCAN,
                        per_parameter_lr_modulation = MEMORY_MODEL_PER_LAYER_LEARNED_LR
                        )
                    )
                        for j in range(depths[i])])

                    #norm = norm_layer(embed_dims[i])
                    #setattr(self, f"norm{i + 1}", norm)
                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

        # self.apply(self._init_weights)

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

    def forward(self, x1, x2):
    #def forward(self, x):
        
        #print(x)
        x = torch.cat([x1, x2], 0)
        
        B = x.shape[0] 
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            #norm = getattr(self, f"norm{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                if i == self.conv_stages:
                    x, H, W = patch_embed(x)
                else:
                    x, H, W = patch_embed(x) #B,N,C
                value_residual=(None,None)
                mem_weight_residual=(None,None)
                for blk in block:
                    x , value_residual,mem_weight_residual  = blk(x,value_residual,mem_weight_residual, B, H, W)
                #x = norm(x)
                x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            appearence_features.append(x)
        return appearence_features


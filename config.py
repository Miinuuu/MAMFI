from functools import partial
import torch.nn as nn
from model.MAMFE import * 
from model.Residual_refiner import *
from model.Motion_estimator import *
'''==========Model config=========='''
def init_model_config(F=32, 
                      embed_dims=None
                     ,in_chans=None
                     ,depth=None
                     ,mlp_ratios=None
                     ,num_heads=None
                     ,scales=None
                     ,hidden_dims=None
                     ,refine=None
                    ,norm_layer=None
                     ,WINDOW_SIZE = None
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

                     ,NEURAL_MEM_SEGMENT_LEN = 144                      # set smaller for more granularity for learning rate / momentum etc
                     ,NEURAL_MEM_BATCH_SIZE = 144                     # set smaller to update the neural memory weights more often as it traverses the sequence
                     ,SLIDING_WINDOWS = False
                     ,STORE_ATTN_POOL_CHUNKS = True                   # whether to use attention pooling for chunk derived momentum, per-layer lr mod, decay
                     ,MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
                     ,NEURAL_MEM_WEIGHT_RESIDUAL = True               # learning to accept contributions from the weights of the previous neural mem layer brings about significant improvements. this was improvised and not in the paper, but inspired by the value residual learning free lunch paper
                     ,NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True        # will allow the neural memory to select what layers from which to derive queries / keys / values, effectively allowing it to graft itself to the transformer in any way to be beneficial. this is to address an issue from a phd student who noted that the mem network is learning nothing more than wk @ wv. this also generalizes all possible ways to connect the neural memory to a transformer, a sort of NAS
                     ,USE_ACCELERATED_SCAN = False
                     ,USE_FLEX_ATTN = False
                        
                      ):
    '''This function should not be modified'''

    return { 
        'embed_dims':embed_dims or [F, 2*F, 4*F, 8*F, 16*F],
        'num_heads': num_heads or [8*F//32, 16*F//32],
        'mlp_ratios':mlp_ratios or [4, 4],
        'qkv_bias':True,
        'norm_layer': norm_layer or partial(nn.RMSNorm, eps=1e-6), 
        'depths':depth or [2,2,2,2,2],
        'window_sizes': [WINDOW_SIZE,WINDOW_SIZE] or [8,8],
        'in_chans':in_chans or 3 

        # MAC Config 
        ,'NEURAL_MEMORY_DEPTH' : NEURAL_MEMORY_DEPTH 
        ,'NUM_PERSIST_MEM':NUM_PERSIST_MEM 
        ,'NUM_LONGTERM_MEM': NUM_LONGTERM_MEM  

        ,'NEURAL_MEM_LAYERS' : NEURAL_MEM_LAYERS                            # layers 2, 4, 6 have neural memory, can add more
        ,'NEURAL_MEM_GATE_ATTN_OUTPUT' : NEURAL_MEM_GATE_ATTN_OUTPUT 
        ,'NEURAL_MEM_MOMENTUM' : NEURAL_MEM_MOMENTUM 
        ,'NEURAL_MEM_MOMENTUM_ORDER' : NEURAL_MEM_MOMENTUM_ORDER 
        ,'NEURAL_MEM_QK_NORM' : NEURAL_MEM_QK_NORM 
        ,'NEURAL_MEM_MAX_LR' : NEURAL_MEM_MAX_LR 

        ,'USE_MEM_ATTENTION_MODEL' : USE_MEM_ATTENTION_MODEL
        ,'USE_MEM_SWIGLUMLP_MODEL' : USE_MEM_SWIGLUMLP_MODEL
        ,'USE_MEM_FACMLP_MODEL' : USE_MEM_FACMLP_MODEL
        ,'USE_MEM_GATEDRESMLP_MODEL' : USE_MEM_GATEDRESMLP_MODEL
        ,'USE_MEM_MLP_MODEL' : USE_MEM_MLP_MODEL

        ,'NEURAL_MEM_SEGMENT_LEN' : NEURAL_MEM_SEGMENT_LEN                      # set smaller for more granularity for learning rate / momentum etc
        ,'NEURAL_MEM_BATCH_SIZE' : NEURAL_MEM_BATCH_SIZE                   # set smaller to update the neural memory weights more often as it traverses the sequence
        ,'SLIDING_WINDOWS' : SLIDING_WINDOWS 
        ,'STORE_ATTN_POOL_CHUNKS' : STORE_ATTN_POOL_CHUNKS                    # whether to use attention pooling for chunk derived momentum, per-layer lr mod, decay
        ,'MEMORY_MODEL_PER_LAYER_LEARNED_LR' : MEMORY_MODEL_PER_LAYER_LEARNED_LR 
        ,'NEURAL_MEM_WEIGHT_RESIDUAL' : NEURAL_MEM_WEIGHT_RESIDUAL              # learning to accept contributions from the weights of the previous neural mem layer brings about significant improvements. this was improvised and not in the paper, but inspired by the value residual learning free lunch paper
        ,'NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW' : NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW        # will allow the neural memory to select what layers from which to derive queries / keys / values, effectively allowing it to graft itself to the transformer in any way to be beneficial. this is to address an issue from a phd student who noted that the mem network is learning nothing more than wk @ wv. this also generalizes all possible ways to connect the neural memory to a transformer, a sort of NAS
        ,'USE_ACCELERATED_SCAN' : USE_ACCELERATED_SCAN 
        ,'USE_FLEX_ATTN' :  USE_FLEX_ATTN  

    }, {
        'embed_dims': embed_dims or [F, 2*F, 4*F, 8*F, 16*F],
        'depths':depth or [2, 2, 2, 2, 2],
        'num_heads':num_heads or [8*F//32, 16*F//32],
        'scales': scales or [8,16],
        'hidden_dims':hidden_dims or [4*F,4*F],
        'c':F,
        'refine': refine,
    }



def Model_create(model=None):


    if model in ['Ours-S'] :
            F=16
            MODEL_CONFIG = {
            'LOGNAME': model ,
            'BASE':None,
            'find_unused_parameters':True,
            'MODEL_TYPE': (MAMFE, Motion_estimator),
            'MODEL_ARCH': init_model_config(
                F = F
                ,depth = [2, 2, 2, 2, 2]
                ,hidden_dims=[4*F,4*F]
                ,refine=Residual_refiner
                ,NEURAL_MEMORY_DEPTH = 1
                ,NUM_PERSIST_MEM = 4
                ,NUM_LONGTERM_MEM = 4
                ,NEURAL_MEM_LAYERS = (1,)                            # layers 2, 4, 6 have neural memory, can add more
                ,NEURAL_MEM_GATE_ATTN_OUTPUT = True
                ,NEURAL_MEM_MOMENTUM = True
                ,NEURAL_MEM_MOMENTUM_ORDER = 1
                ,NEURAL_MEM_QK_NORM = True
                ,NEURAL_MEM_MAX_LR = 1e-1

                ,USE_MEM_ATTENTION_MODEL = False
                ,USE_MEM_SWIGLUMLP_MODEL = False
                ,USE_MEM_FACMLP_MODEL = False
                ,USE_MEM_GATEDRESMLP_MODEL = False
                ,USE_MEM_MLP_MODEL = True

                ,WINDOW_SIZE = 16 
                ,NEURAL_MEM_SEGMENT_LEN = 4 #neural memory chunck size                     # set smaller for more granularity for learning rate / momentum etc
                ,NEURAL_MEM_BATCH_SIZE =  256 #업데이트 단위                     # set smaller to update the neural memory weights more often as it traverses the sequence
                ,SLIDING_WINDOWS = True
                ,STORE_ATTN_POOL_CHUNKS = True                   # whether to use attention pooling for chunk derived momentum, per-layer lr mod, decay
                ,MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
                ,NEURAL_MEM_WEIGHT_RESIDUAL = True               # learning to accept contributions from the weights of the previous neural mem layer brings about significant improvements. this was improvised and not in the paper, but inspired by the value residual learning free lunch paper
                ,NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True        # will allow the neural memory to select what layers from which to derive queries / keys / values, effectively allowing it to graft itself to the transformer in any way to be beneficial. this is to address an issue from a phd student who noted that the mem network is learning nothing more than wk @ wv. this also generalizes all possible ways to connect the neural memory to a transformer, a sort of NAS
                ,USE_ACCELERATED_SCAN = True
                ,USE_FLEX_ATTN = False
                )
            }
    

    elif model in ['Ours-L'] :
            F=32
            MODEL_CONFIG = {
            'LOGNAME': model ,
            'BASE':None,
            'find_unused_parameters':True,
            'MODEL_TYPE': (MAMFE, Motion_estimator),
            'MODEL_ARCH': init_model_config(
                F = F
                ,depth = [2, 2, 2, 2, 2]
                ,hidden_dims=[4*F,4*F]
                ,refine=Residual_refiner
                ,NEURAL_MEMORY_DEPTH = 1
                ,NUM_PERSIST_MEM = 4
                ,NUM_LONGTERM_MEM = 4
                ,NEURAL_MEM_LAYERS = (1,)                            # layers 2, 4, 6 have neural memory, can add more
                ,NEURAL_MEM_GATE_ATTN_OUTPUT = True
                ,NEURAL_MEM_MOMENTUM = True
                ,NEURAL_MEM_MOMENTUM_ORDER = 1
                ,NEURAL_MEM_QK_NORM = True
                ,NEURAL_MEM_MAX_LR = 1e-1

                ,USE_MEM_ATTENTION_MODEL = False
                ,USE_MEM_SWIGLUMLP_MODEL = False
                ,USE_MEM_FACMLP_MODEL = False
                ,USE_MEM_GATEDRESMLP_MODEL = False
                ,USE_MEM_MLP_MODEL = True

                ,WINDOW_SIZE = 16 
                ,NEURAL_MEM_SEGMENT_LEN = 4 #neural memory chunck size                     # set smaller for more granularity for learning rate / momentum etc
                ,NEURAL_MEM_BATCH_SIZE =  256 #업데이트 단위                     # set smaller to update the neural memory weights more often as it traverses the sequence
                ,SLIDING_WINDOWS = True
                ,STORE_ATTN_POOL_CHUNKS = True                   # whether to use attention pooling for chunk derived momentum, per-layer lr mod, decay
                ,MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
                ,NEURAL_MEM_WEIGHT_RESIDUAL = True               # learning to accept contributions from the weights of the previous neural mem layer brings about significant improvements. this was improvised and not in the paper, but inspired by the value residual learning free lunch paper
                ,NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True        # will allow the neural memory to select what layers from which to derive queries / keys / values, effectively allowing it to graft itself to the transformer in any way to be beneficial. this is to address an issue from a phd student who noted that the mem network is learning nothing more than wk @ wv. this also generalizes all possible ways to connect the neural memory to a transformer, a sort of NAS
                ,USE_ACCELERATED_SCAN = True
                ,USE_FLEX_ATTN = False
                )
            }
  

    return MODEL_CONFIG


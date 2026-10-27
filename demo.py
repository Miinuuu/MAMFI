# Standard library imports
import sys
import math
import argparse
import hashlib
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

# Third-party library imports
import cv2
import torch
import numpy as np
from imageio import mimsave
from torchvision.utils import make_grid
import wandb
from fvcore.nn import flop_count_table, FlopCountAnalysis
from ptflops import get_model_complexity_info

# Append current directory for local module imports
sys.path.append('.')

# Local module imports
from config import *
from Trainer import *
from benchmark.utils.padder import InputPadder
from model.visualize import flow2rgb_tensor, norm
from torch_flops import TorchFLOPsByFX


parser = argparse.ArgumentParser()


parser.add_argument('--model', default='TiFi_CSWIN_PAD_8_bi_1622_m1', type=str)
parser.add_argument('--resume', default='TiFi_CSWIN_PAD_8_bi_1622_m1_150_36.04', type=str, help='resume')
parser.add_argument('--trainer', type=str, default='Model',help='trainer')
parser.add_argument('--size', type=int, nargs=2, default=None, metavar=('W', 'H'), help='resize inputs to W H (default: keep original size)')
parser.add_argument('--scale', type=float, default=0.25,help='img scale')
parser.add_argument('--flops_analysis', action='store_true',default=False,help='run FLOPs analysis instead of interpolation')
parser.add_argument('--strict_model', action='store_true', default=False,help='strict model')
parser.add_argument('--project', type=str, default='demo',help='wandb project name')
parser.add_argument('--wandb', action='store_true', default=False, help='log the run to Weights & Biases')

parser.add_argument('--i0',  required=True, help='first input frame')
parser.add_argument('--i1',  default=None,   help='ground-truth middle frame (optional, for PSNR / error map)')
parser.add_argument('--i2',  required=True, help='second input frame')
parser.add_argument('--out_dir', default='example', help='output directory')



args = parser.parse_args()



TTA=False
MODEL_CONFIG= Model_create(args.model)
exec(args.trainer, globals())
created_class = globals()[args.trainer]
model = created_class(local_rank=-1,MODEL_CONFIG=MODEL_CONFIG)        
ckpt=model.load_checkpoint(args.resume)
model.eval()
model.device()

if args.strict_model :
    assert args.model == ckpt['Model']


run = None
if args.wandb:
    wandb.login()
    log_dir = Path.cwd().absolute() / "wandb_logs" / args.model
    log_dir.mkdir(exist_ok=True, parents=True)
    sha = hashlib.sha256()
    sha.update(str(args.model).encode())
    wandb_id = sha.hexdigest()

    note = str(model.num_param) +'M'
    run = wandb.init(project=args.project,
                    id=  wandb_id, 
                    dir= log_dir,
                    job_type='demo',
                    save_code=True,
                    notes=note,
                    name=args.model,
                    resume='allow')

out_dir = Path(args.out_dir)
out_dir.mkdir(exist_ok=True, parents=True)
name = Path(args.i0).stem
has_gt = args.i1 is not None


down_scale=args.scale
print(f'=========================Start Generating=========================')

def read_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'cannot read image: {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

I0 = read_rgb(args.i0)
I2 = read_rgb(args.i2)
# Without a ground-truth middle frame, fall back to I0 only to keep shapes; PSNR is skipped.
I1 = read_rgb(args.i1) if has_gt else I0.copy()

if args.size != None:
    w,h = args.size[0],args.size[1]
    I0 = cv2.resize(I0,(w,h))#1k
    I1 = cv2.resize(I1,(w,h))#1k
    I2 = cv2.resize(I2,(w,h))

h,w = I0.shape[0],I0.shape[1]
print('img shape',I0.shape)
I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I1_ = (torch.tensor(I1.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

padder = InputPadder(I0_.shape, divisor=32)
I0_p, I2_p = padder.pad(I0_, I2_)
timestep = torch.tensor(0.5).reshape(1, 1, 1).unsqueeze(0).cuda()

if args.flops_analysis :
        
        from ptflops import get_model_complexity_info

        # input_constructor 함수 정의: 모델의 forward에 맞춰 입력 생성
        def input_constructor(input_res):
            # input_res는 보통 (H, W) 형태로 전달되는데, 여기서는 이미지 크기 정보로 활용
            # 첫 번째 입력: 이미지 (배치 크기 1, 3 채널, H x W)
            img0 = torch.randn(1, 3, input_res[0], input_res[1]).to('cuda')
            img1 = torch.randn(1, 3, input_res[0], input_res[1]).to('cuda')
            # 두 번째 입력: 추가 피처 벡터 (배치 크기 1, 10)
            img0, img1 = padder.pad(img0, img1)

            imgs=  (torch.cat((img0,img1),1))
            # 모델의 forward는 두 개의 인자를 받으므로 튜플 또는 딕셔너리로 반환할 수 있습니다.
            return {"imgs": imgs, "timestep": timestep}
            # 또는 return (image, features)


        inp=  (torch.cat((I0_p,I2_p),1))
        net = model.net
        macs, params = get_model_complexity_info(net
                                            ,(1280, 720)
                                            ,input_constructor=input_constructor
                                            ,as_strings=True
                                            ,  backend='pytorch'
                                            ,print_per_layer_stat=True
                                            , verbose=True)


        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
else :
    if down_scale != 1.0 :
        print('hr inference')
        pred,flow,mask= (model.hr_inference(I0_p, I2_p, timestep=timestep,TTA=TTA, down_scale=down_scale,fast_TTA=TTA))
        pred = padder.unpad(pred)
        pred_np=(pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    else:
        print('inference')
        pred,flow,mask= (model.inference(I0_p,I2_p, timestep=timestep, TTA=TTA, fast_TTA=TTA))
        pred = padder.unpad(pred)
        pred_np=(pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)


    flowt0 = flow2rgb_tensor(flow[:,0:2])
    flowt0 = padder.unpad(flowt0)

    flowt1 = flow2rgb_tensor(flow[:,2:4])
    flowt1 = padder.unpad(flowt1)
    mask = padder.unpad(mask)

    AD =  torch.abs(pred-I1_)

    panels = [(timestep*I0_+(1-timestep)*I2_)]
    if has_gt:
        panels += [I1_]
    panels += [pred]
    if has_gt:
        panels += [AD]
    preds=torch.cat(panels + [
                     flowt0,
                     flowt1,
                     mask.repeat(1,3,1,1),
                     ] ,dim=0) # overlay, gt, pred
    
    preds=(make_grid(preds[:,[2,1,0]], nrow=preds.shape[1],value_range=(0,1)))    

    images = [I0[:, :, ::-1], pred_np[:, :, ::-1], I2[:, :, ::-1]]
    mimsave(str(out_dir / f'{name}_out_2x.gif'), images, fps=3)

    pred = pred_np
    #cv2.cvtColor(pred_np[:,:,::-1], cv2.COLOR_BGR2RGB)
    psnr = cv2.PSNR(I1, pred) if has_gt else None

    preds=preds.detach().cpu().numpy().transpose(1, 2, 0)*255
    #preds= preds[:,:,::-1]
    cv2.imwrite(str(out_dir / f'{args.resume}_{w}_{h}_{args.scale}_{name}_preds.png'), preds)


    print(f'=========================Done=========================')
    print(f'Model:{args.resume}  PSNR:{psnr}' if has_gt else f'Model:{args.resume}  outputs: {out_dir}/')


if run is not None:
    run.finish()
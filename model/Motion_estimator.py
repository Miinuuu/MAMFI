import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb as wandb
from .Residual_refiner import *
from .layers import * 
from .visualize import * 
from .warplayer import * 



class Head(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head, self).__init__()
        self.upsample = nn.Sequential(nn.PixelShuffle(2), nn.PixelShuffle(2))
        self.scale = scale
        self.conv = nn.Sequential(
                                  conv(in_planes*2 // (4*4) + in_else, c),
                                  conv(c, c),
                                  conv(c, 5),
                                  )  

    def forward(self, motion_feature, x, flow): # /16 /8 /4
        motion_feature = self.upsample(motion_feature) #/4 /2 /1
        if self.scale != 4:
            x = F.interpolate(x, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False)
        if flow != None:
            if self.scale != 4:
                flow = F.interpolate(flow, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False) * 4. / self.scale
            x = torch.cat((x, flow), 1)
        x = self.conv(torch.cat([motion_feature, x], 1))
        if self.scale != 4:
            x = F.interpolate(x, scale_factor = self.scale // 4, mode="bilinear", align_corners=False)
            flow = x[:, :4] * (self.scale // 4)
        else:
            flow = x[:, :4]
        mask = x[:, 4:5]
        return flow, mask
    
    
class Motion_estimator(nn.Module):
    def __init__(self, backbone, **kargs):
        super().__init__()
        print('Motion_estimator')

        self.refine= kargs['refine']
        self.flow_num_stage = len(kargs['hidden_dims'])
        self.feature_bone = backbone

        self.block = nn.ModuleList([Head( kargs['embed_dims'][-1-i], 
                            kargs['scales'][-1-i], 
                            kargs['hidden_dims'][-1-i],
                            7 if i==0 else 18 )
                            for i in range(self.flow_num_stage)])
    
        if self.refine != None :
            self.unet = (kargs['refine'](int(kargs['c']*2)))

    def warp_features(self, xs, flows):
        B = xs[0].size(0) // 2
        return [
            torch.cat((
                backward_warp(x[:B], f[:, :2]),
                backward_warp(x[B:], f[:, 2:4])
            ), dim=1)
            for x, f in zip(xs, flows)
        ]
    #@torch.jit.script_method
    #@torch.jit.script
    def warp_imgs(self, imgs, flows):
        return [
            torch.cat((
                backward_warp(img[:, :3], f[:, :2]),
                backward_warp(img[:, 3:6], f[:, 2:4])
            ), dim=1)
            for img, f in zip(imgs, flows)
        ]

    #@torch.jit.script_method
    #@torch.jit.script_method
    def u_pyramid_features(self, img, flow, mask, levels=[1, 0.5, 0.25]):
        im = [F.interpolate(img, scale_factor=lv, mode="bilinear", align_corners=False) if lv != 1.0 else img for lv in levels]
        f  = [F.interpolate(flow, scale_factor=lv, mode="bilinear", align_corners=False) * lv if lv != 1.0 else flow for lv in levels]
        m  = [F.interpolate(mask, scale_factor=lv, mode="bilinear", align_corners=False) if lv != 1.0 else mask for lv in levels]
        return im, f, m
    
    def calculate_flow(self, imgs, timestep, af=None, mf=None):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        B = img0.size(0)
        flow, mask = None, None
 
        if (af is None) :
            af = self.feature_bone(img0, img1)

        for i in range(self.flow_num_stage):

            if flow != None:
                warped_img0 = backward_warp(img0, flow[:, :2])
                warped_img1 = backward_warp(img1, flow[:, 2:4])


                t=torch.ones_like(imgs[:,:1],device=imgs.device)*timestep.to(device=imgs.device).type(imgs.dtype)

                flow_, mask_ = self.block[i](
                    torch.cat([af[-1-i][:B],af[-1-i][B:]],1),
                    torch.cat((img0, img1, warped_img0, warped_img1, mask,t), 1),
                    flow
                    )

                flow = flow + flow_
                mask = mask + mask_
            else:


                t=torch.ones_like(imgs[:,:1],device=imgs.device)*timestep.to(device=imgs.device).type(imgs.dtype)

                flow, mask = self.block[i](
                    torch.cat([af[-1-i][:B],af[-1-i][B:]],1),
                    torch.cat((img0, img1,t), 1),
                    None
                    )

        return flow, mask
    
    def calculate_flow_hr(self, imgs, timestep, af=None, mf=None):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        B = img0.size(0)
        flow, mask = None, None

        if (af is None) :
            af = self.feature_bone(img0, img1)

        for i in range(self.flow_num_stage):

            if flow != None:
                warped_img0 = backward_warp(img0, flow[:, :2])
                warped_img1 = backward_warp(img1, flow[:, 2:4])


                t=torch.ones_like(imgs[:,:1],device=imgs.device)*timestep.to(device=imgs.device).type(imgs.dtype)

                flow_, mask_ = self.block[i](
                    torch.cat([af[-1-i][:B],af[-1-i][B:]],1),
                    torch.cat((img0, img1, warped_img0, warped_img1, mask,t), 1),
                    flow
                    )

                flow = flow + flow_
                mask = mask + mask_
            else:

                t=torch.ones_like(imgs[:,:1],device=imgs.device)*timestep.to(device=imgs.device).type(imgs.dtype)
                flow, mask = self.block[i](
                    torch.cat([af[-1-i][:B],af[-1-i][B:]],1),
                    torch.cat((img0, img1, t), 1),
                    None
                    )
                
        return flow, mask, af
    
    def calculate_multi_t(self, imgs, step=32, af=None, mf=None):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        B = img0.size(0)
        flow, mask = None, None

        flow_list_t0 ,flow_list_t1 ,mask_list=[],[],[]

        if (af is None) :
            af = self.feature_bone(img0, img1)

        for timestep in torch.linspace(1e-6, 1.0, steps=step, device='cuda') :
            timestep=torch.tensor(timestep).reshape(1, 1, 1).unsqueeze(0).cuda()
            
            print(timestep)
            flow=None
            for i in range(self.flow_num_stage):

                if flow != None:
                    warped_img0 = backward_warp(img0, flow[:, :2])
                    warped_img1 = backward_warp(img1, flow[:, 2:4])

                    t=torch.ones_like(imgs[:,:1],device=imgs.device)*timestep.to(device=imgs.device).type(imgs.dtype)

                    flow_, mask_ = self.block[i](
                        torch.cat([af[-1-i][:B],af[-1-i][B:]],1),
                        torch.cat((img0, img1, warped_img0, warped_img1, mask,t), 1),
                        flow
                        )

                    flow = flow + flow_
                    mask = mask + mask_
                else:
        
                    t=torch.ones_like(imgs[:,:1],device=imgs.device)*timestep.to(device=imgs.device).type(imgs.dtype)
                    flow, mask = self.block[i](
                        torch.cat([af[-1-i][:B],af[-1-i][B:]],1),
                        torch.cat((img0, img1, t), 1),
                        None
                        )
            
            flow_list_t0.append(flow[:,:2].unsqueeze(2))
            flow_list_t1.append(flow[:,2:4].unsqueeze(2))
            mask_list.append(mask.unsqueeze(2))

        flowt0  = torch.cat(flow_list_t0,dim=2) # B,2,t,H,W
        flowt1  = torch.cat(flow_list_t1,dim=2)# B,2,t,H,W
        mask    = torch.cat(mask_list,dim=2)# B,1,t,H,W

        return  flowt0, flowt1,mask, af , None

    def coraseWarp_and_Refine(self, imgs, af, flow, mask):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        warped_img0 = backward_warp(img0, flow[:, :2])
        warped_img1 = backward_warp(img1, flow[:, 2:4])

        if self.refine :

            dimgs,dflows,dmasks= self.u_pyramid_features(imgs,flow,mask,[1.0,0.5,0.25])
            dwimgs = self.warp_imgs(dimgs, dflows)
            Cs = self.warp_features(af[:-2], dflows)
            res=(self.unet(dimgs, dwimgs, dmasks, dflows, Cs)*2)-1
     
            mask_ = torch.sigmoid(mask)
            merged = warped_img0 * mask_ + warped_img1 * (1 - mask_)
            pred = torch.clamp(merged + res,0, 1)

        else:

            mask_ = torch.sigmoid(mask)
            merged = warped_img0 * mask_ + warped_img1 * (1 - mask_)
            pred = torch.clamp(merged , 0, 1)

        return pred


    def forward(self, imgs, timestep=0.5):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        B = imgs.size(0)
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
    
        af = self.feature_bone(img0, img1)
       
        for i in range(self.flow_num_stage):
            if flow != None:

                t=torch.ones_like(img0[:,:1],device=img0.device)*timestep.to(device=img0.device)
                flow_d, mask_d = self.block[i]( torch.cat([af[-1-i][:B],af[-1-i][B:]],1), 
                                                    torch.cat((img0, img1, warped_img0, warped_img1, mask,t), 1), flow)
                flow = flow + flow_d
                mask = mask + mask_d
    
            else:
                t=torch.ones_like(img0[:,:1],device=img0.device)*timestep.to(device=img0.device)
                flow, mask = self.block[i]( torch.cat([af[-1-i][:B],af[-1-i][B:]],1), 
                                                torch.cat((img0, img1,t), 1), None)

            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = backward_warp(img0, flow[:, :2])
            warped_img1 = backward_warp(img1, flow[:, 2:4])
            merged.append(warped_img0 * mask_list[-1] + warped_img1 * (1-mask_list[-1]))
            
        if self.refine :
 
            dimgs,dflows,dmasks= self.u_pyramid_features(imgs,flow,mask,[1.0,0.5,0.25])
            dwimgs = self.warp_imgs(dimgs, dflows)
            Cs = self.warp_features(af[:-self.flow_num_stage], dflows)
            res=(self.unet(dimgs, dwimgs, dmasks, dflows, Cs)*2)-1
            pred = torch.clamp(merged[-1]+res , 0, 1)
        else:
            pred = torch.clamp(merged[-1] , 0, 1)
        
        return flow_list, mask_list, merged, pred
import torch
from einops import rearrange
from PIL import Image, ImageOps
import flow_vis
import os
import cv2
from torchvision import transforms
from torchvision.utils import make_grid



def flow2rgb_tensor(flow_map):
    
    b ,c,h, w = flow_map.shape
    
    rgb_map = torch.ones((b,3,h, w),device=flow_map.device)

    max,_= torch.max(torch.abs(flow_map.view(b,-1)),dim=1)

    normalized_flow_map = flow_map / max.view(b,1, 1, 1)
    
    rgb_map[:,0,:, :] += normalized_flow_map[:, 0]
    rgb_map[:,1,:, :] -= 0.5 * (normalized_flow_map[:, 0] + normalized_flow_map[:, 1])
    rgb_map[:,2,:, :] += normalized_flow_map[:, 1]
    return rgb_map.clip(0, 1)


def norm(data):
        b=data.shape[0]
        max,_=  torch.max((data.view(b,-1)),dim=1)
        min,_=  torch.min((data.view(b,-1)),dim=1)
        '''for i in range(data.dim()-1):
                max=max.unsqueeze(-1)
                min=min.unsqueeze(-1)           '''

        target_shape = max.shape + (1,) * (data.dim() - 1)

        # max와 min 텐서를 한 번에 확장
        max = max.view(target_shape)
        min = min.view(target_shape)

        return (data-min) / (max-min)


def visualize_attn_vanilla(attn0,attn1,H,W):
        attn0=attn0[0].mean(0).reshape(H*W,H*W,1)
        attn1=attn1[0].mean(0).reshape(H*W,H*W,1)

        I0='example/Beanbags/frame11.png'
        I2='example/Beanbags/frame12.png'

        I0 = cv2.imread(I0)
        I2 = cv2.imread(I2) 

        I0=cv2.cvtColor(I0, cv2.COLOR_BGR2RGB)
        I2=cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)

        I0 = cv2.resize(I0,(W,H))#1k
        I2 = cv2.resize(I2,(W,H))


        root ='./attn0/'+str(H)
        root2 ='./attn1/'+str(H)

        if not os.path.exists(root):
            os.makedirs(root)

        if not os.path.exists(root2):
            os.makedirs(root2)

        #print(attn0.shape)
        for i in range (H*W) :
            attn00=attn0[ i:i+1 , : , : ].reshape(H,W,1).detach().cpu().numpy()
            attn11=attn1[i:i+1,:,:].reshape(H,W,1).detach().cpu().numpy()
            attn00=norm(attn00)
            attn11=norm(attn11)
                
            PIL_palette0 = transforms.ToPILImage()(attn00*255).convert('L')
            PIL_palette1 = transforms.ToPILImage()(attn11*255).convert('L')
            #PIL_palette = transforms.ToPILImage()(bbb).convert('RGB')
            PIL_palette0 = ImageOps.colorize(PIL_palette0, '#00ff00', '#ff0000')  # 시작 색상 및 종료 색상을 지정합니다.
            PIL_palette1 = ImageOps.colorize(PIL_palette1, '#00ff00', '#ff0000')  # 시작 색상 및 종료 색상을 지정합니다.
            #PIL_palette.save("./attn/aa/{}.png".format(str(i)))
            PIL_palette0 = Image.blend(PIL_palette0,  Image.fromarray(I0), alpha=0.45)
            PIL_palette1 = Image.blend(PIL_palette1,  Image.fromarray(I2), alpha=0.45)

            red = torch.zeros(1, H*W)
            red[:,i:i+1]=1
            red = rearrange(red, 'r (h w) -> r h w', h=H, w=W)
            #red[:, i:i+1, i:i+1] = 1
            #red = red.view(1, w1, 1, w2, 1).expand(-1, -1, H//w1, -1, W//w2).reshape(1, H,W).contiguous()
            cur_location = torch.cat((red, torch.zeros(2, H, W)), dim=0)
            #cur_location2= cur_location.permute(1,2,0).detach().cpu().numpy()*255
            cur_location = transforms.ToPILImage()(cur_location).convert('RGB')
            
            topk0 = Image.blend(PIL_palette0, cur_location, alpha=0.5)
            topk1 = Image.blend(PIL_palette1, cur_location, alpha=0.5)
            topk0.save("./attn0/{}/{}.png".format(str(H),str(i)))
            topk1.save("./attn1/{}/{}.png".format(str(H),str(i)))
            
            #bb= attn00*I2
            #bb= cv2.addWeighted(bb,0.5,cur_location2,0.5,0)
            #bb=cv2.resize(bb,(256,256))
            #cv2.imwrite( root2+'/'+str(i)+'.png',cv2.cvtColor(bb, cv2.COLOR_BGR2RGB))






def visualize_attn_swin(attn0,attn1,H,W, I0='example/Beanbags/frame11.png',I2='example/Beanbags/frame12.png'):
  

        B_NW,NH,WS,_ = attn0.shape
        w1= int(WS ** (1/2))
        w2=w1
        NW=H//w1 * W//w2
        B= B_NW//NW

        attn0=attn0.reshape(B,NW,NH,WS,WS)
        attn1=attn1.reshape(B,NW,NH,WS,WS)
        attn0=attn0[0].mean(1).reshape(NW,WS,WS,1)
        attn1=attn1[0].mean(1).reshape(NW,WS,WS,1)

        I0 = cv2.imread(I0)
        I2 = cv2.imread(I2) 

        I0=cv2.cvtColor(I0, cv2.COLOR_BGR2RGB)
        I2=cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)

        I0 = cv2.resize(I0,(W,H))#1k
        I2 = cv2.resize(I2,(W,H))


        root ='./attn0/'+str(H)
        root2 ='./attn1/'+str(H)
        root3 ='./attn2/'+str(H)
        if not os.path.exists(root):
            os.makedirs(root)

        if not os.path.exists(root2):
            os.makedirs(root2)

        if not os.path.exists(root3):
                os.makedirs(root3)
        #print(attn0.shape)

        for i in range(WS):
            red= torch.zeros(NW,WS)
            attn00=attn0[ : , i:i+1 , : , : ].reshape(NW,w1,w2,1)
            attn11=attn1[ : , i:i+1 , : , : ].reshape(NW,w1,w2,1)
            for j in range (NW) :
                attn00=norm(attn00)
                attn11=norm(attn11)
                #red = torch.zeros(1, WS)
                red[j:j+1,i:i+1]=1
                #red = rearrange(red, 'r (h w) -> r h w', h=w1, w=w2)

            attn00=attn00.reshape(H//w1,W//w2,w1,w2,1).permute(0,2,1,3,4).reshape(H,W,1).detach().cpu().numpy() #w1,w2,1
            attn11=attn11.reshape(H//w1,W//w2,w1,w2,1).permute(0,2,1,3,4).reshape(H,W,1).detach().cpu().numpy()
            PIL_palette0 = transforms.ToPILImage()(attn00*255).convert('L')
            PIL_palette1 = transforms.ToPILImage()(attn11*255).convert('L')
            #PIL_palette = transforms.ToPILImage()(bbb).convert('RGB')
            PIL_palette0 = ImageOps.colorize(PIL_palette0, '#00ff00', '#ff0000')  # 시작 색상 및 종료 색상을 지정합니다.
            PIL_palette1 = ImageOps.colorize(PIL_palette1, '#00ff00', '#ff0000')  # 시작 색상 및 종료 색상을 지정합니다.
            #PIL_palette.save("./attn/aa/{}.png".format(str(i)))
            PIL_palette0 = Image.blend(PIL_palette0,  Image.fromarray(I0), alpha=0.45)
            PIL_palette1 = Image.blend(PIL_palette1,  Image.fromarray(I2), alpha=0.45)

            red=red.reshape(H//w1,W//w2,w1,w2).permute(0,2,1,3).reshape(1,H,W)
            cur_location = torch.cat((red, torch.zeros(2, H, W)), dim=0)
            cur_location2= cur_location.permute(1,2,0).detach().cpu().numpy()*255 #
            cur_location = transforms.ToPILImage()(cur_location).convert('RGB')
            topk0 = Image.blend(PIL_palette0, cur_location, alpha=0.5)
            topk1 = Image.blend(PIL_palette1, cur_location, alpha=0.5)
            topk0.save("./attn0/{}/{}.png".format(str(H),str(i)))
            topk1.save("./attn1/{}/{}.png".format(str(H),str(i)))
            
            bb= attn00*I2
            bb= cv2.addWeighted(bb,0.5,cur_location2,0.5,0)
            bb=cv2.resize(bb,(256,256))
            cv2.imwrite( root3+'/'+str(i)+'.png',cv2.cvtColor(bb, cv2.COLOR_BGR2RGB))








def visualize_hidden_state(attn0,attn1,H,W, I0='example/Beanbags/frame07.png',I2='example/Beanbags/frame09.png'):
  

        NW,c,w1,w2 = attn0.shape

        #print(NW, H//w1 * W//w2)
        #assert NW == H//w1 * W//w2

        attn0=attn0.mean(1).reshape(NW,w1,w2,1)
        attn1=attn1.mean(1).reshape(NW,w1,w2,1)

        I0 = cv2.imread(I0)
        I2 = cv2.imread(I2) 

        I0=cv2.cvtColor(I0, cv2.COLOR_BGR2RGB)
        I2=cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)

        I0 = cv2.resize(I0,(W,H))#1k
        I2 = cv2.resize(I2,(W,H))


        root ='./hidden_state0/'+str(H)
        root2 ='./hidden_state1/'+str(H)
        root3 ='./hidden_state2/'+str(H)

        if not os.path.exists(root):
            os.makedirs(root)
        if not os.path.exists(root2):
            os.makedirs(root2)
        if not os.path.exists(root3):
            os.makedirs(root3)
        
        
        for j in range (NW) :
                #attn00=norm(attn0[j])
                #attn11=norm(attn1[j])
                attn00=(attn0[j])
                attn11=(attn1[j])


                #red = torch.zeros(1, WS)
                # red[j:j+1,i:i+1]=1
                #red = rearrange(red, 'r (h w) -> r h w', h=w1, w=w2)

                attn00=attn00.detach().cpu().numpy() #w1,w2,1
                attn11=attn11.detach().cpu().numpy()
                PIL_palette0 = transforms.ToPILImage()(attn00*255).convert('L')
                PIL_palette1 = transforms.ToPILImage()(attn11*255).convert('L')
                #PIL_palette = transforms.ToPILImage()(bbb).convert('RGB')
                PIL_palette0 = ImageOps.colorize(PIL_palette0, '#001100', '#ff0000')  # 시작 색상 및 종료 색상을 지정합니다.
                PIL_palette1 = ImageOps.colorize(PIL_palette1, '#001100', '#ff0000')  # 시작 색상 및 종료 색상을 지정합니다.
                #PIL_palette.save("./attn/aa/{}.png".format(str(i)))
                PIL_palette0 = Image.blend(PIL_palette0,  Image.fromarray(I0), alpha=0.2)
                PIL_palette1 = Image.blend(PIL_palette1,  Image.fromarray(I2), alpha=0.2)

                # red=red.reshape(H//w1,W//w2,w1,w2).permute(0,2,1,3).reshape(1,H,W)
                # cur_location = torch.cat((red, torch.zeros(2, H, W)), dim=0)
                # cur_location2= cur_location.permute(1,2,0).detach().cpu().numpy()*255 #
                # cur_location = transforms.ToPILImage()(cur_location).convert('RGB')
                # topk0 = Image.blend(PIL_palette0, cur_location, alpha=0.5)
                # topk1 = Image.blend(PIL_palette1, cur_location, alpha=0.5)

                PIL_palette0.save("./hidden_state0/{}/{}.png".format(str(H),str(j)))
                PIL_palette1.save("./hidden_state1/{}/{}.png".format(str(H),str(j)))
                
                #bb= attn00*I2
                #bb= cv2.addWeighted(bb,0.5,cur_location2,0.5,0)
                #bb=cv2.resize(bb,(256,256))
                # cv2.imwrite( root3+'/'+str(j)+'.png',cv2.cvtColor(PIL_palette0, cv2.COLOR_BGR2RGB))






def visualize_qkv_swin(q,k,v,H,W) :
        
        B_NW,NH,WS,C_NH=q.shape
        w1= int(WS**(1/2))
        w2= w1
        C= C_NH * NH
        NW= H//w1 * W//w2
        B= B_NW//NW

        
        qqq = q.permute(0,2,1,3).reshape(B_NW,w1,w2,C)
        qqq=qqq.view(B ,H//w1, W//w2, w1,w2,C)
        qqq=qqq.permute(0,1,3,2,4,5)
        qqq=qqq.reshape(B,H,W,C)
        qqq=qqq[0:1, :  ,:, : ].squeeze().detach().cpu().numpy()

        kkk = k.permute(0,2,1,3).reshape(B_NW,w1,w2,C)
        kkk=kkk.view(B ,H//w1, W//w2, w1,w2,C)
        kkk=kkk.permute(0,1,3,2,4,5)
        kkk=kkk.reshape(B,H,W,C)
        kkk=kkk[0:1, :  ,:, : ].squeeze().detach().cpu().numpy()

        vvv = v.permute(0,2,1,3).reshape(B_NW,w1,w2,C)
        vvv=vvv.view(B ,H//w1, W//w2, w1,w2,C)
        vvv=vvv.permute(0,1,3,2,4,5)
        vvv=vvv.reshape(B,H,W,C)
        vvv=vvv[0:1, :  ,:, : ].squeeze().detach().cpu().numpy()

        qqq=norm(qqq)*255
        kkk=norm(kkk)*255
        vvv=norm(vvv)*255

        root0 ='./query/'+str(H)
        root1 ='./key/'+str(H)
        root2 ='./value/'+str(H)

        if not os.path.exists(root0):
                os.makedirs(root0)

        if not os.path.exists(root1):
                os.makedirs(root1)

        if not os.path.exists(root2):
                os.makedirs(root2)
                
        for i in range (C) :
            cv2.imwrite( root0+'/'+str(i)+'.png',qqq[...,i])
            cv2.imwrite( root1+'/'+str(i)+'.png',kkk[...,i])
            cv2.imwrite( root2+'/'+str(i)+'.png',vvv[...,i])
 

def visualize_qkv_vanilla(q,k,v,H,W) :

        B,NH,N,C_NH = q.shape
        C = C_NH*NH

        qqq= q.permute(0,2,1,3).reshape(B,H,W,C)[0:1, :  ,:, : ].squeeze().detach().cpu().numpy()
        kkk= k.permute(0,2,1,3).reshape(B,H,W,C)[0:1, :  ,:, : ].squeeze().detach().cpu().numpy()
        kkk= v.permute(0,2,1,3).reshape(B,H,W,C)[0:1, :  ,:, : ].squeeze().detach().cpu().numpy()


        qqq=norm(qqq)*255
        kkk=norm(kkk)*255
        vvv=norm(vvv)*255

        root0 ='./query/'+str(H)
        root1 ='./key/'+str(H)
        root2 ='./value/'+str(H)

        if not os.path.exists(root0):
                os.makedirs(root0)

        if not os.path.exists(root1):
                os.makedirs(root1)

        if not os.path.exists(root2):
                os.makedirs(root2)

        for i in range (C) :
            cv2.imwrite( root0+'/'+str(i)+'.png',qqq[...,i])
            cv2.imwrite( root1+'/'+str(i)+'.png',kkk[...,i])
            cv2.imwrite( root2+'/'+str(i)+'.png',vvv[...,i])
 

def visualize_fm_swin(v,H,W,root2='./af0') :
        
        B_NW,WS,MD=v.shape
        w1= int(WS**(1/2))
        w2= w1
        NW= H//w1 * W//w2
        B= B_NW//NW

        v = v.reshape(B,NW,w1,w2,MD)
        v = v.reshape(B,H//w1,W//w2,w1,w2,MD)
        v = v.permute(0,1,3,2,4,5).reshape(B,H,W,MD)
        v=norm(v)*255
        v=v.detach().cpu().numpy()

        for j in range (B) :
                root0 =str(root2)+'/'+str(j)
                if not os.path.exists(root0):
                        os.makedirs(root0)
                for i in range (MD) :
                        cv2.imwrite( root0+'/'+str(H)+'_'+str(W)+'_'+str(i)+'.png',(v[j,...,i]))
                        #cv2.imwrite( root0+'/'+str(H)+'_'+str(W)+'_'+str(i)+'.png',norm(v[j,...,i])*255)

def visualize_dwt_fm_swin(v,NW,H,W,root2='./af0') :
        
        BNW,c,w0,w1 = v.shape
        
        B=BNW//NW
        v=v.reshape(BNW//NW,NW,c,w0,w1)

        v = v.permute(0,1,3,4,2)
        v = v.reshape(BNW//NW,H,W,c)
        v=norm(v)*255
        v=v.detach().cpu().numpy()

        for j in range (B) :
                root0 =str(root2)+'/'+str(j)
                if not os.path.exists(root0):
                        os.makedirs(root0)
                for i in range (c) :
                        cv2.imwrite( root0+'/'+str(H)+'_'+str(W)+'_'+str(i)+'.png',v[j,...,i])


def visualize_flow(flow,root2 ='./flow/left/0') :
        B,CH,H,W= flow.shape
        np_flow0=flow.permute(0,2,3,1)# b , h , w, c
        np_flow0 = np_flow0.detach().cpu().numpy()

        for j in range (B):
                root= root2+'/'+str(j)
                if not os.path.exists(root):
                        os.makedirs(root)
                flow_color = flow_vis.flow_to_color(np_flow0[j,...], convert_to_bgr=False)

            #print(np_flow0.shape)
                cv2.imwrite( root+'/'+ str(H)+'_'+str(W) +'flow'+ '.png',flow_color)


def visualize_dfm_swin(v,root2='./af/df0') :
        
        B,C,H,W=v.shape
        v=v.permute(0,2,3,1)
        v=norm(v)*255
        v=v.detach().cpu().numpy()

        for j in range (B) :
                root0 =str(root2)+'/'+str(j)
                if not os.path.exists(root0):
                        os.makedirs(root0)
                for i in range (C) :
                        cv2.imwrite( root0+'/'+str(H)+'_'+str(W)+'_'+str(i)+'.png',v[j,...,i])

def visualize_grid(v, size,root='./output/grid/',filename='pred.png' ) :
        
        if not os.path.exists(root):
                os.makedirs(root)
        B,C,H,W=size
        preds = torch.cat(v,dim=1)
        preds=(make_grid(preds.reshape(-1,3,H,W)[:,[2,1,0]], nrow=preds.shape[1],value_range=(0,1)))    

        preds= preds.permute(1,2,0)*255
        cv2.imwrite( os.path.join(root,filename,), preds.detach().cpu().numpy() )
               

def visualize_flow(flow,root2 ='./flow/left/') :
        B,CH,H,W= flow.shape
        np_flow0=flow.permute(0,2,3,1)# b , h , w, c
        np_flow0 = np_flow0.detach().cpu().numpy()

        for j in range (B):
                root= root2+'/'+str(j)
                print(root)
                if not os.path.exists(root):
                        os.makedirs(root)
                flow_color = flow_vis.flow_to_color(np_flow0[j,...], convert_to_bgr=False)

            #print(np_flow0.shape)
                cv2.imwrite( root+'/'+ str(H)+'_'+str(W) +'flow'+ '.png',flow_color)

def visualize_flow2(flow,root2 ='./flow/left/0') :
        B,CH,H,W= flow.shape
        np_flow0=flow.permute(0,2,3,1)# b , h , w, c
        np_flow0 = np_flow0.detach().cpu().numpy()

        for j in range (B):
                for k in range (CH//2):
                        root= root2+'/'+str(j)
                        if not os.path.exists(root):
                                os.makedirs(root)
                        flow_color = flow_vis.flow_to_color(np_flow0[j,...,k*2:k*2+2], convert_to_bgr=False)

                #print(np_flow0.shape)
                        cv2.imwrite( root+'/'+ str(H)+'_'+str(W) +'_'+str(k)+'flow'+'.png',flow_color)

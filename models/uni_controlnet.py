import einops
import torch

from einops import rearrange, repeat
from torchvision.utils import make_grid

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import torch.nn.functional as F
from pytorch_msssim import ssim
from models_geo.models import Unified, Grid2d
import torch.nn as nn
#from ldm.models.diffusion.ddpm import DDPMSampler
#插值函数
# def interpolate_tensor(input_tensor):
#     assert len(input_tensor.shape) == 6
#     import pdb; pdb.set_trace()
#     # reshape the tensor to 5D
#     b, d1, d2, d3, h, w = input_tensor.shape
#     input_tensor = input_tensor.reshape(-1, h, w).unsqueeze(1)  # [b*d1*d2*d3, 1, h, w]

#     # interpolate
#     output_tensor = F.interpolate(input_tensor, size=(256, 1024), mode='bilinear', align_corners=False)

#     # reshape the tensor back to 6D
#     output_tensor = output_tensor.squeeze(1).reshape(b, d1, d2, d3, 256, 1024)  # [b, d1, d2, d3, 256, 1024]

#     return output_tensor
def interpolate_tensor(input_tensor, batch_size=4):
    assert len(input_tensor.shape) == 6
    #import pdb; pdb.set_trace()
    b, d1, d2, d3, h, w = input_tensor.shape
    output_tensor = torch.empty(b, d1, d2, d3, 256, 1024, device=input_tensor.device, dtype=input_tensor.dtype)

    for i in range(0, b, batch_size):
        batch = input_tensor[i:i+batch_size]
        batch = batch.reshape(-1, h, w).unsqueeze(1)  # [b*d1*d2*d3, 1, h, w]
        interpolated_batch = F.interpolate(batch, size=(256, 1024), mode='bilinear', align_corners=False)
        interpolated_batch = interpolated_batch.squeeze(1).reshape(-1, d1, d2, d3, 256, 1024)  # [b, d1, d2, d3, 256, 1024]
        output_tensor[i:i+batch_size] = interpolated_batch

    return output_tensor

class CNN_AttentionReduce(nn.Module):
    def __init__(self, output_channels):
        super(CNN_AttentionReduce, self).__init__()
        self.conv = nn.Conv2d(32*32, output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # shape: [B, 32, 32, H, W]
        B, _, _, H, W = x.shape

        # Reshape to [B, 32*32, H, W]
        x = x.view(B, 32*32, H, W)

        # Apply 1x1 convolution
        x = self.conv(x)

        return x

class CNN_AttentionReduce_3layers(nn.Module):
    def __init__(self, output_channels):
        super(CNN_AttentionReduce_3layers, self).__init__()
        
        # First 1x1 convolution to reduce channels
        self.conv1 = nn.Conv2d(32*32, 512, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolution layer
        self.conv2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)

        # Third convolution layer to get the desired output channels
        self.conv3 = nn.Conv2d(256, output_channels, kernel_size=1, stride=1, padding=0)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        # shape: [B, 32, 32, H, W]
        B, _, _, H, W = x.shape

        # Reshape to [B, 32*32, H, W]
        x = x.view(B, 32*32, H, W)

        # Apply the convolutions with ReLU activations
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))

        return x
    
class CNN_FeatureReduce_sat(nn.Module):
    def __init__(self, input_channels=256):
        super(CNN_FeatureReduce_sat, self).__init__()
        # 卷积层：输入通道数为 input_channels，输出通道数为 1
        self.conv = nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x)

class CNN_FeatureReduce_sat_3layers(nn.Module):
    def __init__(self, input_channels=256):
        super(CNN_FeatureReduce_sat_3layers, self).__init__()
        
        # First 1x1 convolution
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)

        # Second 1x1 convolution
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)

        # Third 1x1 convolution to get the desired output channel (1 channel)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply the convolutions with ReLU activations
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)

        return x
    
class UniControlNet(LatentDiffusion):

    def __init__(self, mode, local_control_config=None, global_control_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mode in ['local', 'global', 'uni']
        self.mode = mode
        if self.mode in ['local', 'uni']:
            self.local_adapter = instantiate_from_config(local_control_config)
            self.local_control_scales = [1.0] * 13
        if self.mode in ['global', 'uni']:
            self.global_adapter = instantiate_from_config(global_control_config)
        self.mask = True  #自己加mask控制
        context = {
            x: True for x in ["distance", "orientation", "panorama", "overhead"]
            }
        self.geonet = Unified(num_output=13,context=context)
        self.geogrid = Grid2d(context)

        #进行提取出来的特征维度缩减
        # 调整网络结构
        #self.CNN_AttentionReduce = CNN_AttentionReduce(output_channels=1)
        self.avg_attention_reduce = False
        #self.CNN_FeatureReduce_sat = CNN_FeatureReduce_sat()
        self.CNN_AttentionReduce = CNN_AttentionReduce_3layers(output_channels=1)
        self.CNN_FeatureReduce_sat = CNN_FeatureReduce_sat_3layers()

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        #import pdb; pdb.set_trace()
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)#[4,4,32,128],[4,77,768]


        # add geo attention data
        t_image = batch['t_image']
        label = batch['label']
        bbox = batch['bbox']
        near_locs = batch['near_locs']
        near_images = batch['near_images']

        if len(batch['local_conditions']) != 0:
            local_conditions = batch['local_conditions']
            if bs is not None:
                local_conditions = local_conditions[:bs]
            local_conditions = local_conditions.to(self.device)
            local_conditions = einops.rearrange(local_conditions, 'b h w c -> b c h w')
            local_conditions = local_conditions.to(memory_format=torch.contiguous_format).float()
        else:
            local_conditions = torch.zeros(1,1,1,1).to(self.device).to(memory_format=torch.contiguous_format).float() #[B,12,256,1024]
        if len(batch['global_conditions']) != 0:
            global_conditions = batch['global_conditions']
            if bs is not None:
                global_conditions = global_conditions[:bs]
            global_conditions = global_conditions.to(self.device).to(memory_format=torch.contiguous_format).float()
        else:
            global_conditions = torch.zeros(1,1).to(self.device).to(memory_format=torch.contiguous_format).float()

        return x, dict(c_crossattn=[c], local_control=[local_conditions], global_control=[global_conditions],
                       t_image=t_image, label=label,bbox=bbox,near_locs=near_locs, near_images=near_images) # add geo data

    def apply_model(self, x_noisy, t, cond, global_strength=1, *args, **kwargs):  #这里隐空间，可以改
        # TODO soft mask要在这里改？pixel level condition
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        #import pdb; pdb.set_trace()
        if self.mode in ['global', 'uni']:
            assert cond['global_control'][0] != None
            global_control = self.global_adapter(cond['global_control'][0])
            cond_txt = torch.cat([cond_txt, global_strength*global_control], dim=1)
        if self.mode in ['local', 'uni']:
            assert cond['local_control'][0] != None
            local_control = torch.cat(cond['local_control'], 1)  #[B,12,256,1024]
            
            #if len(cond.keys())>3:  #加geo attention soft mask


            #output, attention, sat_attention = self.geonet(cond['t_image'], cond['bbox'], cond['near_locs'], cond['near_images'])
            # attention: [B,32,32, 20, 16, 64] 32是grid size   sat_attention: [B, 256, 32, 32] 
            #sat_attention, attention_street = self.geogrid(c1['bbox'], c1['near_locs'], c1['near_images'],c1['t_image'])
            #生成的attention mask插值
            #在插值前就设阈值
            #attn_mask = interpolate_tensor(attention[:,:,:,0,:,:])

            local_control = self.local_adapter(x=x_noisy, timesteps=t, context=cond_txt, local_conditions=local_control) #list, len=13
            local_control = [c * scale for c, scale in zip(local_control, self.local_control_scales)]  #这里加soft mask
        
        if self.mode == 'global':
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt)
        else:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, local_control=local_control)
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)
    
    @torch.no_grad()
    def log_images(self, batch, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, plot_denoise_rows=False,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, **kwargs):
        use_ddim = ddim_steps is not None
        N = batch['jpg'].shape[0]  #batch size
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)

        # add geo attention
        #import pdb; pdb.set_trace()
        c1=c
        
        if self.mask:  # attention should be [4,32,32,20,8,32]
            output, attention, sat_attention = self.geonet(c1['t_image'], c1['bbox'], c1['near_locs'], c1['near_images'])
            #sat_attention, attention_street = self.geogrid(c1['bbox'], c1['near_locs'], c1['near_images'],c1['t_image'])
            #生成的attention mask插值
            #在插值前就设阈值
            #attn_mask = interpolate_tensor(attention[:,:,:,0,:,:])
            #import pdb; pdb.set_trace()
            if self.CNN_AttentionReduce:
                att_map1 = self.CNN_AttentionReduce(attention[:,:,:,1,:,:])
                attn_mask1 = F.interpolate(att_map1, size=(256, 1024), mode='bilinear', align_corners=False).squeeze(1)
                att_map2 = self.CNN_AttentionReduce(attention[:,:,:,2,:,:])
                attn_mask2 = F.interpolate(att_map2, size=(256, 1024), mode='bilinear', align_corners=False).squeeze(1)
                attn_satmap = self.CNN_FeatureReduce_sat(sat_attention)
                sat_mask = F.interpolate(attn_satmap,size=(256, 256), mode='bilinear', align_corners=False)
                sat_mask = sat_mask.repeat(1,1,1,4)
                geo_mask=torch.cat((sat_mask,attn_mask1.unsqueeze(1),attn_mask2.unsqueeze(1)),dim=1)
            else:
                att_map1 = attention[:,0,0,1,:,:]
                attn_mask1 = F.interpolate(att_map1.unsqueeze(1), size=(256, 1024), mode='bilinear', align_corners=False).squeeze(1)
                att_map2 = attention[:,0,0,2,:,:]
                attn_mask2 = F.interpolate(att_map2.unsqueeze(1), size=(256, 1024), mode='bilinear', align_corners=False).squeeze(1)
                attn_satmap = sat_attention[:,0,:,:]
                sat_mask = F.interpolate(attn_satmap.unsqueeze(1),size=(256, 256), mode='bilinear', align_corners=False)
                sat_mask = sat_mask.repeat(1,1,1,4)
            #atten_mask = F.interpolate(attention[0,0,0,0,:,:],size=(256,1024), mode='bilinear',align_corners=False)

        c_cat = c["local_control"][0][:N]
        c_global = c["global_control"][0][:N]
        c = c["c_crossattn"][0][:N]
        #import pdb; pdb.set_trace()
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["local_control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)
        # add mask images threshold=0.5
        #print(c["local_control"][0].shape)
        #import pdb; pdb.set_trace()
        #使用 hard mask
        #log['near1_masked'] = c_cat[:,6:9,:,:]*(attn_mask1.unsqueeze(1)>attn_mask1.mean())
        #log['near2_masked'] = c_cat[:,9:12,:,:]*(attn_mask2.unsqueeze(1)>attn_mask2.mean())
        #或者使用soft mask
        log['near1_masked'] = c_cat[:,6:9,:,:]*attn_mask1.unsqueeze(1).repeat(1,3,1,1)
        log['near2_masked'] = c_cat[:,9:12,:,:]*attn_mask2.unsqueeze(1).repeat(1,3,1,1)
        
        # log[]
        # log['sat_mask'] = 
        #log['near1_mask'] = attn_mask1.unsqueeze(1)
        #log['near2_mask'] = attn_mask2.unsqueeze(1)
        #log['sat_mask'] = sat_mask.unsqueeze(1)
        

        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:  #不走
            samples, z_denoise_row = self.sample_log(cond={"local_control": [c_cat], "c_crossattn": [c], "global_control": [c_global]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps,  eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid
        

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat
            uc_global = torch.zeros_like(c_global)
            uc_full = {"local_control": [uc_cat], "c_crossattn": [uc_cross], "global_control": [uc_global],"mask1": [attn_mask1],"mask2":[attn_mask2],"sat_mask":[sat_mask]}
            
            samples_cfg, _ = self.sample_log(cond={"local_control": [c_cat], "c_crossattn": [c], "global_control": [c_global],"mask1": [attn_mask1],"mask2":[attn_mask2],"sat_mask":[sat_mask]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg  #生成的图片

        return log
    # def test_step(self, batch, batch_idx):
    #     psnr = calculate_psnr(images['reconstruction'], x_samples_cfg)
    #     ssim = calculate_ssim(images['reconstruction'], x_samples_cfg)
    #     self.log('psnr', psnr)
    #     self.log('ssim', ssim)
    #     return psnr,ssim
    
    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        
        if self.mode == 'global':
            h, w = 512, 512
        else:
            _, _, h, w = cond["local_control"][0].shape  #原尺寸
        shape = (self.channels, h // 8, w // 8)
        #import pdb; pdb.set_trace()
        if self.mask:
            context = {
            x: True for x in ["distance", "orientation", "panorama", "overhead"]
            }
            ''' hard mask
            mask1 = cond["mask1"][0] >cond["mask1"][0].mean()  # near_pano0
            mask2 = cond["mask2"][0] >cond["mask2"][0].mean()  # near_pano1
            mask_sat = cond["sat_mask"][0] >cond["sat_mask"][0].mean()
            # mask1 = cond["mask1"][0] #>cond["mask1"][0].mean()  # near_pano0
            # mask2 = cond["mask2"][0] #>cond["mask2"][0].mean()  # near_pano1
            # mask_sat = cond["sat_mask"][0] #>cond["mask_sat"][0].mean()
            mask = torch.cat((mask1,mask2),dim=0).unsqueeze(1)
            #先用hard mask试？soft mask需要加到guidance里面，设置对应的guidance系数
            cond['local_control'][0][:,6:9,:,:] = cond['local_control'][0][:,6:9,:,:]* mask1.unsqueeze(1)
            cond['local_control'][0][:,10:12,:,:] = cond['local_control'][0][:,10:12,:,:]* mask2.unsqueeze(1)
            '''
            #简单的soft mask
            #import pdb; pdb.set_trace()
            cond['local_control'][0][:,6:9,:,:] = cond['local_control'][0][:,6:9,:,:]* cond["mask1"][0].unsqueeze(1).repeat(1,3,1,1)
            cond['local_control'][0][:,9:12,:,:] = cond['local_control'][0][:,9:12,:,:]* cond["mask2"][0].unsqueeze(1).repeat(1,3,1,1)
            mask = torch.cat((cond["mask1"][0],cond["mask2"][0]),dim=0).unsqueeze(1)
            # cond['local_control'][0][:,12,:,:] = mask1.unsqueeze(1)  #near view1 attention
            # cond['local_control'][0][:,13,:,:] = mask2.unsqueeze(1)  #near view2 attention
            # cond['local_control'][0][:,14,:,:] = mask_sat.unsqueeze(1)  #sat attention
            
            # cond['local_control'][0] = torch.cat((cond['local_control'][0],mask1.unsqueeze(1)),dim=1)
            # cond['local_control'][0] = torch.cat((cond['local_control'][0],mask2.unsqueeze(1)),dim=1)
            # cond['local_control'][0] = torch.cat((cond['local_control'][0],mask_sat),dim=1)
            #用concat
            
             
        #import pdb; pdb.set_trace()
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, mask=mask, verbose=False, **kwargs) #加了mask
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        if self.mode in ['local', 'uni']:
            params += list(self.local_adapter.parameters())
        if self.mode in ['global', 'uni']:
            params += list(self.global_adapter.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            if self.mode in ['local', 'uni']:
                self.local_adapter = self.local_adapter.cuda()
            if self.mode in ['global', 'uni']:
                self.global_adapter = self.global_adapter.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            if self.mode in ['local', 'uni']:
                self.local_adapter = self.local_adapter.cpu()
            if self.mode in ['global', 'uni']:
                self.global_adapter = self.global_adapter.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()


def calculate_psnr(img1, img2):
    mse_loss = F.mse_loss(img1, img2, reduction='none').mean(dim=(1,2,3))
    psnr = 10 * ((1**2)/mse_loss).log10()
    return psnr.mean()

def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=1, size_average=True)
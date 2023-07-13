import os

import numpy as np
from PIL import Image
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
import torchmetrics
#
#from ldm.modules.image_degradation.utils_image import calculate_psnr,calculate_ssim

import torch.nn.functional as F
from pytorch_msssim import ssim
from pytorch_fid import fid_score
#from torchmetrics.image.fid import FrechetInceptionDistance

class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, num_local_conditions=7):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.num_local_conditions = num_local_conditions
        #self.FID = FID()

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            if k == 'local_control':
                _, _, h, w = images[k].shape
                if h == w == 1:
                    continue
                #import pdb; pdb.set_trace()
                for local_idx in range(self.num_local_conditions):
                    grid = torchvision.utils.make_grid(images[k][:, 3*local_idx: 3*(local_idx+1), :, : ], nrow=4)
                    if self.rescale:
                        grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    grid = grid.numpy()
                    grid = (grid * 255).astype(np.uint8)
                    filename = "gs-{:06}_e-{:06}_b-{:06}_{}_{}.png".format(global_step, current_epoch, batch_idx, k, local_idx)
                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    Image.fromarray(grid).save(path)
            elif k != 'global_control':
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "gs-{:06}_e-{:06}_b-{:06}_{}.png".format(global_step, current_epoch, batch_idx, k)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        #import pdb; pdb.set_trace()
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)
            

            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            #['reconstruction', 'local_control', 'conditioning', 'samples_cfg_scale_9.00']
            generated_images = []
            gt_images = []
            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)  # 进Unicontrolnet, 返回的就是四张图

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)
            # add metrics
            # log metrics
            #import pdb; pdb.set_trace()
            metrics = {"clip_score": {'things': [], 'laion-art': [], 'CC3M':[]}, 'edge_rmse': [], 'delta_clip_score': {'things': [], 'laion-art':[], 'CC3M':[]}, 'aesthetics_score': [], 'sampling_time': np.round(np.mean(sampling_times), 2)}
            generated_images.append(images['samples_cfg_scale_9.00'])
            gt_images.append(images['reconstruction'])
            try:  # sometimes the calculation fails with "sqrtm: array must not contain infs or NaNs"
                metrics['fid_score'] = fid_score.calculate_fid(generated_images, gt_images, batch_size=32, 
                    device='cuda' if torch.cuda.is_available() else 'cpu', dims=2048, num_workers=4)
            except:
                pass

            if is_train:
                pl_module.train()
            # add return?
            if split == 'test':
                #pl_module.test()
                
                return images

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        #import pdb; pdb.set_trace()
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
    # add test
        else:
            images = {}
            images = self.log_img(pl_module, batch, batch_idx, split="test")
            psnr = calculate_psnr(images['reconstruction'], images['samples_cfg_scale_9.00'])
            ssim = calculate_ssim(images['reconstruction'], images['samples_cfg_scale_9.00'])
            print('PSNR:', psnr, '  ssim:', ssim)
            
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        #import pdb; pdb.set_trace()
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="test")
        else:
            images = {}
            images = self.log_img(pl_module, batch, batch_idx, split="test")
            psnr = calculate_psnr(images['reconstruction'], images['samples_cfg_scale_9.00'])
            ssim = calculate_ssim(images['reconstruction'], images['samples_cfg_scale_9.00'])
            FID = self.FID(images['reconstruction'], images['samples_cfg_scale_9.00'])

            print('PSNR:', psnr, ' ssim:', ssim, 'FID', FID)


def calculate_psnr(img1, img2):
    mse_loss = F.mse_loss(img1, img2, reduction='none').mean(dim=(1,2,3))
    psnr = 10 * ((1**2)/mse_loss).log10()
    return psnr.mean()

def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=1, size_average=True)
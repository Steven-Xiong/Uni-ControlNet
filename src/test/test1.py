import sys
if './' not in sys.path:
    sys.path.append('./')
from utils.share import *
import utils.config as config
from omegaconf import OmegaConf
import argparse

import cv2
import einops
import gradio as gr
import numpy as np
import os

import torch
from pytorch_lightning import seed_everything

from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.mlsd import MLSDdetector
from annotator.hed import HEDdetector
from annotator.sketch import SketchDetector
from annotator.openpose import OpenposeDetector
from annotator.midas import MidasDetector
from annotator.uniformer import UniformerDetector
from annotator.content import ContentDetector

from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler
from ldm.util import instantiate_from_config
from torch.utils.data import DataLoader

from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
from pytorch_msssim import ssim
from pytorch_fid import fid_score
import torchvision.transforms as transforms
from PIL import Image

from scipy.spatial import distance as dist
from scipy.stats import wasserstein_distance

parser = argparse.ArgumentParser(description='Uni-ControlNet testing')
parser.add_argument('--config-path', type=str, default='./configs/test_v15.yaml')
parser.add_argument('--learning-rate', type=float, default=2e-5)
parser.add_argument('---batch-size', type=int, default=32)
parser.add_argument('---training-steps', type=int, default=1e5)
parser.add_argument('---resume-path', type=str, default='log_local/lightning_logs/Brooklyn/epoch=61-step=50000.ckpt')
parser.add_argument('---logdir', type=str, default='./log_local/')
parser.add_argument('---log-freq', type=int, default=500)
parser.add_argument('---sd-locked', type=bool, default=True)
parser.add_argument('---num-workers', type=int, default=16)
parser.add_argument('---gpus', type=int, default=1)
# self
parser.add_argument('--mode', default = 'local')
parser.add_argument('--ddim_steps', type= int, default = 50)
parser.add_argument('--input_path', type=str, default = 'log_local/test/input_brooklyn')
parser.add_argument('--output_path', type= str, default = 'log_local/test/output_brooklyn')

args = parser.parse_args()


def calculate_psnr(img1, img2):
    mse_loss = F.mse_loss(img1, img2, reduction='none').mean(dim=(1,2,3))
    psnr = 10 * ((1**2)/mse_loss).log10()
    return psnr.mean()

def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=1, size_average=True)

def calculate_sd_score(img1, img2, bins=(8, 8, 8)):
    # 首先确认两个图像的尺寸是否相同
    assert img1.shape == img2.shape, "Both images must be of the same shape"
    
    # 将图像数据的范围从 [0,1] 转化为 [0,255] 并转化为 uint8 类型
    img1 = (img1 * 255).astype("uint8")
    img2 = (img2 * 255).astype("uint8")
    
    # 计算两个图像的颜色直方图
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()

    hist2 = cv2.calcHist([img2], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # 计算两个颜色直方图的 earth mover's distance
    return wasserstein_distance(hist1, hist2)

if __name__ == '__main__':
    apply_canny = CannyDetector()
    apply_mlsd = MLSDdetector()
    apply_hed = HEDdetector()
    apply_sketch = SketchDetector()
    apply_openpose = OpenposeDetector()
    apply_midas = MidasDetector()
    apply_seg = UniformerDetector()
    apply_content = ContentDetector()


    # model = create_model('./configs/uni_v15.yaml').cpu()
    # model.load_state_dict(load_state_dict('./ckpt/uni.ckpt', location='cuda'))
    # model = model.cuda()
    # ddim_sampler = DDIMSampler(model)

    config_path = args.config_path
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    training_steps = args.training_steps
    resume_path = args.resume_path
    default_logdir = args.logdir
    logger_freq = args.log_freq
    sd_locked = args.sd_locked
    num_workers = args.num_workers
    gpus = args.gpus
    mode = args.mode
    ddim_steps = args.ddim_steps
    
    # a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
    # n_prompt = gr.Textbox(label="Negative Prompt",
    # value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
    
    a_prompt = 'best quality, extremely detailed'
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    # content embedding空置
    content_emb = np.zeros((768))
    
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config['model'])

    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    #import pdb; pdb.set_trace()
    dataset = instantiate_from_config(config['data'])
    device = 'cuda'
    
    model.to(device)
    ddim_sampler = DDIMSampler(model)
    
    test_dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=True, shuffle=True)
    #指标计算：
    psnr_scores = []
    ssim_scores = []
    rmse_scores = []
    sd_scores = []
    results_list = []
    input_list = []
    print('len: ', len(test_dataloader))
    for batch_idx, data in enumerate(test_dataloader):
        #for debug 
        # if batch_idx>=10:
        #     break
        target = data['jpg'].to(device, dtype = torch.float)
        prompt = data['txt'][0]
        local_conditions = data['local_conditions'].to(device, dtype = torch.float)
        local_conditions = einops.rearrange(local_conditions, 'b h w c -> b c h w').clone()
        # global_conditions = data['global_conditions'].to(device)
        # global_conditions = einops.rearrange(global_conditions, 'b h w c -> b c h w').clone()
        global_control = torch.from_numpy(content_emb.copy()).float().cuda().clone()
        global_control = torch.stack([global_control for _ in range(batch_size)], dim=0)
        
        #target, prompt, local_conditions, global_conditions = data #[x.to(device) for x in data]
        uc_local_control = local_conditions
        uc_global_control = torch.zeros_like(global_control)
        #import pdb; pdb.set_trace()
        cond = {"local_control": [local_conditions], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * batch_size)], 'global_control': [uc_global_control]}
        un_cond = {"local_control": [uc_local_control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * batch_size)], 'global_control': [uc_global_control]}
        if mode == 'global':
            h, w = 512, 512
        else:
            _, _, h, w = cond["local_control"][0].shape
        shape = (4, h // 8, w // 8)
        samples, _ = ddim_sampler.sample(ddim_steps, batch_size,
                                                        shape, cond, verbose=False, eta=0.00,
                                                        unconditional_guidance_scale=7.50,
                                                        unconditional_conditioning=un_cond, global_strength=1)
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(batch_size)]
        
        target = target.cpu().numpy()
        #import pdb; pdb.set_trace()
        for i in range(batch_size):
            input_list.append(target[i])
        for i in range(batch_size):
            results_list.append((x_samples[i].astype('float32')/255.0))
            # input_list.append(target[i] for i in range(batch_size))
            # results_list.append(x_samples[i] for i in range(batch_size))
        
    #import pdb; pdb.set_trace()
    print(len(input_list), len(results_list))
    #rmse = np.sqrt(mean_squared_error(input_list, results_list))
    for img1, img2 in zip(input_list, results_list):
        
        psnr_tmp = compute_psnr(img1,img2)
        ssim_tmp = compute_ssim(img1,img2, channel_axis=-1, data_range=1.0) #或者ssim_tmp = compute_ssim(img1*255,img2*255, channel_axis=-1, data_range=255)
        mse = np.mean((img1 - img2) ** 2)
        rmse_tmp = np.sqrt(mse)
        sd_tmp = calculate_sd_score(img1, img2)
        
        psnr_scores.append(psnr_tmp)
        ssim_scores.append(ssim_tmp)
        rmse_scores.append(rmse_tmp)
        sd_scores.append(sd_tmp)
        with open('evaluate_brooklyn_psnr.txt','a') as f:
            f.write(str(psnr_tmp)+'\n')
        with open('evaluate_brooklyn_ssim.txt','a') as f:
            f.write(str(ssim_tmp)+'\n')
        with open('evaluate_brooklyn_rmse.txt','a') as f:
            f.write(str(rmse_tmp)+'\n')
        with open('evaluate_brooklyn_sd.txt','a') as f:
            f.write(str(sd_tmp)+'\n')
    
    #import pdb; pdb.set_trace()
    psnr_score = np.mean(psnr_scores)
    ssim_score = np.mean(ssim_scores)
    rmse_score = np.mean(rmse_scores)
    sd_score = np.mean(sd_scores)
    #ssim_val = ssim( input_list, results_list, data_range=255, size_average=False)
    
    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    for i, img_tensor in enumerate(input_list):
        #cv2.imwrite(os.path.join(input_path,('input'+str(i)+'.png')), img_tensor*255)
        transforms.ToPILImage()(torch.from_numpy(img_tensor.transpose(2,0,1))).save(os.path.join(input_path,(str(i)+'.png')))
    for i, img_tensor in enumerate(results_list):
        #cv2.imwrite(os.path.join(output_path,('output'+str(i)+'.png')), img_tensor*255)
        transforms.ToPILImage()(torch.from_numpy(img_tensor.transpose(2,0,1))).save(os.path.join(output_path,(str(i)+'.png')))
    #import pdb; pdb.set_trace()
        
    fid = fid_score.calculate_fid_given_paths([input_path, output_path], 
                                                    batch_size=50, 
                                                    device=device, 
                                                    dims=2048)
    print('PSNR: ', psnr_score, 'SSIM: ', ssim_score, 'FID: ', fid, 'RMSE: ', rmse_score, 'SD: ', sd_score)
    with open('evaluate_brooklyn.txt','a') as f:
        f.write('PSNR: '+ str(psnr_score))
        f.write('\nSSIM: '+ str(ssim_score))
        f.write('\nRMSE: '+ str(rmse_score))
        f.write('\nFID: '+ str(fid))
        f.write('\nSD: '+ str(sd_score))
        
    
    
  

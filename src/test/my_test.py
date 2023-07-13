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

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

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

from ldm.modules.image_degradation.utils_image import calculate_psnr,calculate_ssim
from ldm.util import instantiate_from_config

from models.logger import ImageLogger
from pytorch_lightning.callbacks import ModelCheckpoint
#import cvp

# apply_canny = CannyDetector()
# apply_mlsd = MLSDdetector()
# apply_hed = HEDdetector()
# apply_sketch = SketchDetector()
# apply_openpose = OpenposeDetector()
# apply_midas = MidasDetector()
# apply_seg = UniformerDetector()
# apply_content = ContentDetector()


# model = create_model('./configs/test_v15.yaml').cpu()
# model.load_state_dict(load_state_dict('./ckpt/uni.ckpt', location='cuda'))
# model = model.cuda()
# ddim_sampler = DDIMSampler(model)


def process(canny_image, mlsd_image, hed_image, sketch_image, openpose_image, midas_image, seg_image, content_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta, low_threshold, high_threshold, value_threshold, distance_threshold, alpha, global_strength):
    
    seed_everything(seed)

    if canny_image is not None:
        anchor_image = canny_image
    elif mlsd_image is not None:
        anchor_image = mlsd_image
    elif hed_image is not None:
        anchor_image = hed_image
    elif sketch_image is not None:
        anchor_image = sketch_image
    elif openpose_image is not None:
        anchor_image = openpose_image
    elif midas_image is not None:
        anchor_image = midas_image
    elif seg_image is not None:
        anchor_image = seg_image
    elif content_image is not None:
        anchor_image = content_image
    else:
        anchor_image = np.zeros((image_resolution, image_resolution, 3)).astype(np.uint8)
    H, W, C = resize_image(HWC3(anchor_image), image_resolution).shape

    with torch.no_grad():
        if canny_image is not None:
            canny_image = cv2.resize(canny_image, (W, H))
            canny_detected_map = HWC3(apply_canny(HWC3(canny_image), low_threshold, high_threshold))
        else:
            canny_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        if mlsd_image is not None:
            mlsd_image = cv2.resize(mlsd_image, (W, H))
            mlsd_detected_map = HWC3(apply_mlsd(HWC3(mlsd_image), value_threshold, distance_threshold))
        else:
            mlsd_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        if hed_image is not None:
            hed_image = cv2.resize(hed_image, (W, H))
            hed_detected_map = HWC3(apply_hed(HWC3(hed_image)))
        else:
            hed_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        if sketch_image is not None:
            sketch_image = cv2.resize(sketch_image, (W, H))
            sketch_detected_map = HWC3(apply_sketch(HWC3(sketch_image)))            
        else:
            sketch_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        if openpose_image is not None:
            openpose_image = cv2.resize(openpose_image, (W, H))
            openpose_detected_map, _ = apply_openpose(HWC3(openpose_image), False)
            openpose_detected_map = HWC3(openpose_detected_map)
        else:
            openpose_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        if midas_image is not None:
            midas_image = cv2.resize(midas_image, (W, H))
            midas_detected_map = HWC3(apply_midas(HWC3(midas_image), alpha))
        else:
            midas_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        if seg_image is not None:
            seg_image = cv2.resize(seg_image, (W, H))
            seg_detected_map, _ = apply_seg(HWC3(seg_image))
            seg_detected_map = HWC3(seg_detected_map)
        else:
            seg_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        if content_image is not None:
            content_emb = apply_content(content_image)
        else:
            content_emb = np.zeros((768))

        detected_maps_list = [canny_detected_map, 
                              mlsd_detected_map, 
                              hed_detected_map,
                              sketch_detected_map,
                              openpose_detected_map,
                              midas_detected_map,
                              seg_detected_map                          
                              ]
        detected_maps = np.concatenate(detected_maps_list, axis=2)

        local_control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
        local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)
        local_control = einops.rearrange(local_control, 'b h w c -> b c h w').clone()
        global_control = torch.from_numpy(content_emb.copy()).float().cuda().clone()
        global_control = torch.stack([global_control for _ in range(num_samples)], dim=0)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        uc_local_control = local_control
        uc_global_control = torch.zeros_like(global_control)
        cond = {"local_control": [local_control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], 'global_control': [global_control]}
        un_cond = {"local_control": [uc_local_control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)], 'global_control': [uc_global_control]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength] * 13
        samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond, global_strength=global_strength)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(num_samples)]
    #import pdb; pdb.set_trace()
    return [results, detected_maps_list]


parser = argparse.ArgumentParser(description='Uni-ControlNet Training')
parser.add_argument('--config-path', type=str, default='./configs/test_v15.yaml')
parser.add_argument('--learning-rate', type=float, default=2e-5)
parser.add_argument('---batch-size', type=int, default=8)
parser.add_argument('---training-steps', type=int, default=1e5)
parser.add_argument('---resume-path', type=str, default='log_local/lightning_logs/epoch=61-step=50000.ckpt')
parser.add_argument('---logdir', type=str, default='./log_local/')
parser.add_argument('---log-freq', type=int, default=500)
parser.add_argument('---sd-locked', type=bool, default=True)
parser.add_argument('---num-workers', type=int, default=16)
parser.add_argument('---gpus', type=int, default=1)
args = parser.parse_args()

# modify based on Coming downtoearth
if __name__ == '__main__':

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

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config['model'])

    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked

    #model.eval()
    

    # model = cvp.CVP.load_from_checkpoint(
    #   '{}/lightning_logs/version_0/checkpoints/last.ckpt'.format(job_dir))
    # model.to(device)
    # model.eval()
    #import pdb; pdb.set_trace()
    dataset = instantiate_from_config(config['data'])
    #dataset.getitem(1)
    test_dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=True, shuffle=True)
    
    logger = ImageLogger(batch_frequency=logger_freq, num_local_conditions =2, disabled = True)
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=logger_freq,
    )

    trainer = pl.Trainer(
        gpus=gpus,
        callbacks=[logger, checkpoint_callback], 
        default_root_dir=default_logdir,
        max_steps=training_steps,
    )
    trainer.test(model,
        test_dataloader
    )
    #trainer.test(model, dataloaders = test_dataloader)
    # device = 'cuda'
    # for batch_idx, data in enumerate(test_dataloader):
    #    target, prompt, local_conditions, global_conditions = [x.to(device) for x in data]
       
       



class CVP(pl.LightningModule):

  def __init__(self, **kwargs):
    super().__init__()

    self.save_hyperparameters()
    #self.config = brooklynqueensdataset(self.hparams.dataset, 'train').config
    self.net = models.build_model(self.hparams.method, self.config)

  def forward(self, image, bbox, near_locs, near_images):
    return self.net(image, bbox, near_locs, near_images)

  def loss_function(self, output, target):
    if self.config["loss"] == "cross_entropy":
      mask = torch.ones_like(target)
      for item in self.config['ignore_label']:
        mask[target == item] = 0

      valid_inds = torch.nonzero(mask, as_tuple=True)
      b, h, w = valid_inds

      return torch.nn.functional.cross_entropy(output[b, :, h, w],
                                               target[b, h, w].long())
    elif self.config["loss"] == "huber":
      return torch.nn.functional.smooth_l1_loss(output, target.unsqueeze(1))
    elif self.config["loss"] == "uncertainty":
      mean = output[:, 0, :, :]
      log_var = output[:, 1, :, :]
      first = .5 * torch.exp(-log_var) * F.mse_loss(
          mean, target, reduction="none")
      second = .5 * log_var
      return torch.mean(first + second)
    else:
      raise NotImplementedError

  def training_step(self, batch, batch_idx):
    image, label, bbox, near_locs, near_images = batch
    output = self(image, bbox, near_locs, near_images)
    loss = self.loss_function(output, label)
    self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    image, label, bbox, near_locs, near_images = batch
    output = self(image, bbox, near_locs, near_images)
    loss = self.loss_function(output, label)
    self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)

    if batch_idx == 0:
      if self.config["loss"] == "cross_entropy":
        pred = torch.true_divide(
            torch.argmax(output.cpu(), dim=1, keepdim=True),
            self.config["num_output"] - 1)
      elif self.config["loss"] == "huber":
        pred = output
      elif self.config["loss"] == "uncertainty":
        pred = output[:, 0, :, :].unsqueeze(1)

      image_grid = torchvision.utils.make_grid(image.cpu())
      pred_grid = torchvision.utils.make_grid(pred.cpu())
      self.logger.experiment.add_image('image', image_grid, self.current_epoch)
      self.logger.experiment.add_image('pred', pred_grid, self.current_epoch)

    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(),
                                 lr=self.hparams.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=1,
                                                gamma=0.96)
    return [optimizer], [scheduler]

  def train_dataloader(self):
    print(" [*] loading data from '%s'" % self.hparams.dataset)
    dataset = BrooklynQueensDataset(self.hparams.dataset, 'train')
    return DataLoader(dataset,
                      batch_size=self.hparams.batch_size,
                      shuffle=True,
                      num_workers=12)

  def val_dataloader(self):
    print(" [*] loading data from '%s'" % self.hparams.dataset)
    dataset = BrooklynQueensDataset(self.hparams.dataset, 'val')
    return DataLoader(dataset,
                      batch_size=self.hparams.batch_size,
                      shuffle=False,
                      num_workers=12)
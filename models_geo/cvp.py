"""
  Author: S
"""

import torch
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader

import models
from data import BrooklynQueensDataset


class CVP(pl.LightningModule):

  def __init__(self, **kwargs):
    super().__init__()

    self.save_hyperparameters()
    self.config = BrooklynQueensDataset(self.hparams.dataset, 'train').config
    #import pdb; pdb.set_trace()
    self.net = models.build_model(self.hparams, self.config)

  def forward(self, image, bbox, near_locs, near_images):
    output, _ = self.net(image, bbox, near_locs, near_images)
    return output

  def forward_attention(self, image, bbox, near_locs, near_images):
    output, attention = self.net(image, bbox, near_locs, near_images)
    return output, attention

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
    output = self(image, bbox, near_locs, near_images)  #shape:[4,13,256,256]
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
                      num_workers=24)

  def val_dataloader(self):
    print(" [*] loading data from '%s'" % self.hparams.dataset)
    dataset = BrooklynQueensDataset(self.hparams.dataset, 'val')
    return DataLoader(dataset,
                      batch_size=self.hparams.batch_size,
                      shuffle=False,
                      num_workers=24)

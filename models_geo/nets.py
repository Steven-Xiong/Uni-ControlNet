import numpy as np
import sys
# if './' not in sys.path:
# 	sys.path.append('./')
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .efficient import EfficientNet
from models_geo.grid import *



class BaseImage(nn.Module):

  def __init__(self):
    super().__init__()

    self.encoder = Encoder_geo()

  def forward(self, image):
    return self.encoder(image)


class BaseGround(nn.Module):

  def __init__(self, out_channels=256, frozen=False):
    super().__init__()

    self.frozen = frozen

    resnet = models.resnet50(pretrained=True)
    self.encoder = nn.Sequential(*list(resnet.children())[:-3])
    self.ln = nn.LayerNorm([256, 16, 64]) #【128,8,32】

    for idx, (name, param) in enumerate(self.encoder.named_parameters()):
      #print(idx, name, param.shape)
      if self.frozen:
        param.requires_grad = False
      elif idx <= 71:
        param.requires_grad = False

    self.reduce = nn.Conv2d(1024, out_channels, 1)

  def forward(self, near_images):
    b, num_neighbors, c, h, w = near_images.shape

    images = torch.reshape(near_images, (b * num_neighbors, c, h, w))

    feats = self.encoder(images)              #encoder提取特征
    feats_reduced = F.relu(self.ln(self.reduce(feats)))

    _, c, h, w = feats_reduced.shape

    near_feats = torch.reshape(feats_reduced, (b, num_neighbors, c, h, w))

    return near_feats


class BaseGrid(nn.Module):

  def __init__(self, context=None):
    super().__init__()

    self.grid = Grid2d(context)
    self.norm = nn.BatchNorm2d(256)  #128

  def forward(self, bbox, near_locs, near_feats, overhead_feat=None):
    #import pdb; pdb.set_trace()
    grid_, attention = self.grid(bbox, near_locs, near_feats, overhead_feat)
    grid = self.norm(grid_.permute(0, 3, 1, 2))
    return grid, attention


class BaseUnified(nn.Module):

  def __init__(self,
               feat_channels=56,
               grid_channels=256,  #128
               out_channels=[160, 448]):
    super().__init__()

    f1_c, f2_c = out_channels

    self.f1_c1 = nn.Conv2d(feat_channels + grid_channels, f1_c, 3, padding=1)
    self.f1_c2 = nn.Conv2d(f1_c, f1_c, 3, padding=1)
    self.f1_c3 = nn.Conv2d(f1_c, f1_c, 3, padding=1)
    self.f1_bn1 = nn.BatchNorm2d(f1_c)
    self.f1_bn2 = nn.BatchNorm2d(f1_c)
    self.f1_bn3 = nn.BatchNorm2d(f1_c)

    self.f2_c1 = nn.Conv2d(f1_c, f2_c, 3, padding=1)
    self.f2_c2 = nn.Conv2d(f2_c, f2_c, 3, padding=1)
    self.f2_c3 = nn.Conv2d(f2_c, f2_c, 3, padding=1)
    self.f2_bn1 = nn.BatchNorm2d(f2_c)
    self.f2_bn2 = nn.BatchNorm2d(f2_c)
    self.f2_bn3 = nn.BatchNorm2d(f2_c)

  def forward(self, feat, grid): 
    fused = torch.cat((feat, grid), 1)

    f1_c1 = F.relu(self.f1_bn1(self.f1_c1(fused)))
    f1_c2 = F.relu(self.f1_bn2(self.f1_c2(f1_c1)))
    f1_c3 = F.max_pool2d(F.relu(self.f1_bn3(self.f1_c3(f1_c2))), 2)

    f2_c1 = F.relu(self.f2_bn1(self.f2_c1(f1_c3)))
    f2_c2 = F.relu(self.f2_bn2(self.f2_c2(f2_c1)))
    f2_c3 = F.max_pool2d(F.relu(self.f2_bn3(self.f2_c3(f2_c2))), 2)

    return [f1_c3, f2_c3]


class Encoder_geo(nn.Module):

  def __init__(self):
    super().__init__()

    self.efficient = EfficientNet.from_name('efficientnet-b4')

  def forward(self, x):
    endpoints = self.efficient.extract_endpoints(x)
    e1 = endpoints['reduction_2']
    e2 = endpoints['reduction_3']
    e3 = endpoints['reduction_4']
    e4 = endpoints['reduction_5']

    return [e1, e2, e3, e4]


class Decoder_geo(nn.Module):

  def __init__(self, num_outputs, filters=[32, 56, 160, 448]):
    super().__init__()

    self.up = nn.Upsample(scale_factor=2, mode='nearest')

    self.decoder5 = DoubleConv(filters[3], filters[3])
    self.decoder4 = DoubleConv(filters[3] + filters[2], filters[2])
    self.decoder3 = DoubleConv(filters[2] + filters[1], filters[1])
    self.decoder2 = DoubleConv(filters[1] + filters[0], filters[1] + filters[0])
    self.decoder1 = DoubleConv(filters[1] + filters[0], filters[1] + filters[0])

    self.finalblock = nn.Conv2d(filters[1] + filters[0], num_outputs, 1)

  def forward(self, encodings):
    e1, e2, e3, e4 = encodings

    d5 = self.decoder5(self.up(e4))
    d5 = torch.cat([d5, e3], dim=1)

    d4 = self.decoder4(self.up(d5))
    d4 = torch.cat([d4, e2], dim=1)

    d3 = self.decoder3(self.up(d4))
    d3 = torch.cat([d3, e1], dim=1)

    d2 = self.decoder2(self.up(d3))

    d1 = self.decoder1(self.up(d2))

    return self.finalblock(d1)


class DoubleConv(nn.Module):

  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
                              nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                              nn.Conv2d(out_ch, out_ch, 3, padding=1),
                              nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

  def forward(self, x):
    x = self.conv(x)
    return x


if __name__ == "__main__":
  target_shape = (256, 256, 3)
  trans = torchvision.transforms.ToTensor()
  im = np.random.uniform(size=target_shape)
  im = trans(im).unsqueeze(0).float()

  pano = torch.tensor(np.random.rand(1, 20, 3, 256, 1024)).float()
  encoder_g = BaseGround()
  e1 = encoder_g(pano)
  print(e1.shape)

  encoder = BaseImage()
  decoder = Decoder_geo(12)
  e1, e2, e3, e4 = encoder(im)
  print(decoder([e1, e2, e3, e4]).shape)

  grid = torch.tensor(np.random.rand(1, 256, 32, 32)).float()
  unified = BaseUnified()
  e3_new, e4_new = unified(e2, grid)
  print(e3_new.shape, e4_new.shape)

  print(decoder([e1, e2, e3_new, e4_new]).shape)

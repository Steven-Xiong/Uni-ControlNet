import torch

from models_geo.nets import *


def build_model(args, config):
  print("[*] building model from {}".format(args.method))
  #import pdb; pdb.set_trace()
  if args.method == "geo-attention":
    model = Unified(config['num_output'], args.context)  #走这里
  elif args.method == "satellite":
    model = Satellite(config['num_output'])
  elif args.method == "ground":
    model = Ground(config['num_output'], args.context)
  elif args.method == "lwa":
    raise NotImplementedError
  elif args.method == "random":
    raise Exception('Method random, no need for training.')
  else:
    raise Exception('Method unrecognized.')

  return model

#主框架
class Unified(nn.Module):

  def __init__(self, num_output, context=None):
    super().__init__()

    self.encoder_g = BaseGround(out_channels=256)  #128

    self.encoder = BaseImage()
    self.grid = BaseGrid(context)
    self.unified = BaseUnified()
    self.decoder = Decoder_geo(num_output)

  def forward(self, image, bbox, near_locs, near_images):
    #import pdb; pdb.set_trace()         # image:[4,3,256,256]
    e1, e2, _, _ = self.encoder(image)  # e1:[B,32,64,64] e2:[4,56,32,32]

    near_feats = self.encoder_g(near_images)  # shape [B, num_nears, 128 ,8,32]

    grid, attention = self.grid(bbox, near_locs, near_feats, e2) # grid: [B,128,32,32]
                                                                 # attention: [B,32,32,num_near,8,32]
    e3, e4 = self.unified(e2, grid)    # e3:[B,160,16,16] e4:[4,448,8,8]

    output = self.decoder([e1, e2, e3, e4])  # output:[4,13,256,256]

    return output, attention, grid  #grid为sat view的attention


class Satellite(nn.Module):

  def __init__(self, num_output):
    super().__init__()

    self.encoder = BaseImage()
    self.decoder = Decoder_geo(num_output)

  def forward(self, image, bbox, near_locs, near_images):
    e1, e2, e3, e4 = self.encoder(image)
    output = self.decoder([e1, e2, e3, e4])

    return output, None


class Ground(nn.Module):

  def __init__(self, num_output, context=None):
    super().__init__()

    if isinstance(context, dict):
      assert ("overhead" not in context.keys())

    self.encoder_g = BaseGround(out_channels=128, frozen=True)

    self.grid = BaseGrid(context)

    # U-Net style decoder
    self.decoder3 = DoubleConv(128, 64)
    self.decoder2 = DoubleConv(64, 64)
    self.decoder1 = DoubleConv(64, 32)  #所以是32的grid

    self.up = nn.Upsample(scale_factor=2, mode='nearest')
    self.finalconv1 = nn.Conv2d(32, 32, 3, padding=1)
    self.finalconv2 = nn.Conv2d(32, num_output, 1)

  def forward(self, image, bbox, near_locs, near_images):
    near_feats = self.encoder_g(near_images)

    grid, attention = self.grid(bbox, near_locs, near_feats)

    d3 = self.decoder3(grid)
    d2 = self.decoder2(self.up(d3))
    d1 = self.decoder1(self.up(d2))

    f1 = F.relu(self.finalconv1(self.up(d1)))
    output = self.finalconv2(f1)

    return output, attention


if __name__ == "__main__":
  im = torch.tensor(np.random.rand(2, 3, 256, 256)).float()  #sat image
  bbox = torch.tensor(np.random.rand(2, 4)).float()
  near_locs = torch.tensor(np.random.rand(2, 20, 2)).float()
  near_images = torch.tensor(np.random.rand(2, 20, 3, 128, 500)).float() #near location的集合

  unified = Unified(12)
  print(unified(im, bbox, near_locs, near_images)[0].shape)

  satellite = Satellite(12)
  print(satellite(im, bbox, near_locs, near_images)[0].shape)

  ground = Ground(12, {"distance": True, "orientation": True})
  print(ground(im, bbox, near_locs, near_images)[0].shape)

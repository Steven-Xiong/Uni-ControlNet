import torch
import torch.nn as nn
import torch.nn.functional as F
from models_geo import ops
#from .ops import *

import numpy as np

from torch import nn, einsum
from einops import rearrange, repeat
#加的辅助函数

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

#以下为Transformer替代
class CrossAttention(nn.Module):
  def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
      super().__init__()
      inner_dim = dim_head * heads
      context_dim = default(context_dim, query_dim)

      self.scale = dim_head ** -0.5
      self.heads = heads

      self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
      self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
      self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

      self.to_out = nn.Sequential(
          nn.Linear(inner_dim, query_dim),
          nn.Dropout(dropout)
      )

  def forward(self, x, context=None, mask=None):
      h = self.heads

      q = self.to_q(x)
      context = default(context, x)
      k = self.to_k(context)
      v = self.to_v(context)

      q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

      sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

      if exists(mask):
          mask = rearrange(mask, 'b ... -> b (...)')
          max_neg_value = -torch.finfo(sim.dtype).max
          mask = repeat(mask, 'b j -> (b h) () j', h=h)
          sim.masked_fill_(~mask, max_neg_value)

      # attention, what we cannot get enough of
      attn = sim.softmax(dim=-1)

      out = einsum('b i j, b j d -> b i d', attn, v)
      out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
      return self.to_out(out)
    
class BasicTransformerBlock(nn.Module):
  def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
      super().__init__()
      self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
      self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
      self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                  heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
      self.norm1 = nn.LayerNorm(dim)
      self.norm2 = nn.LayerNorm(dim)
      self.norm3 = nn.LayerNorm(dim)
      self.checkpoint = checkpoint

  def forward(self, x, context=None):
      return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

  def _forward(self, x, context=None):
      x = self.attn1(self.norm1(x)) + x
      x = self.attn2(self.norm2(x), context=context) + x
      x = self.ff(self.norm3(x)) + x
      return x
      
class Grid2d(nn.Module):

  def __init__(self, context=None, version=3):
    super().__init__()

    if context is None:
      context = {
          x: True for x in ["distance", "orientation", "panorama", "overhead"]
      }
    if version == 3:
      self.grid = GridV3(context)     #use GridV3
    else:
      self.grid = GridV2(context)

  def forward(self, bbox, near_locs, near_feats, overhead_feat):
    return self.grid(bbox, near_locs, near_feats, overhead_feat)


class GridV3(nn.Module):

  def __init__(self, context, grid_size=32):
    super().__init__()

    self.grid_size = grid_size
    self.attention = self.GeoAttention(**context)
    #改自己的transformer:
    #self.attention = self.GeoTransformer(**context)
    self.rays = torch.tensor(ops.binned_rays(h_pano=32, w_pano=64), dtype=torch.float32)

  def forward(self, bbox, near_locs, near_feats, overhead_feat):
    num_batch, num_neighbors, num_feats = near_feats.shape[:3]
    num_locs = self.grid_size**2
    height, width = self.rays.shape[2:]

    grid = []
    attention = []
    for idx in range(num_batch):
      l_bbox = bbox[idx, :]
      l_near_locs = near_locs[idx, ...]
      l_near_feats = near_feats[idx, ...]
      l_overhead_feat = None

      if overhead_feat is not None:
        l_overhead_feat = overhead_feat[idx,
                                        ...].flatten(start_dim=1).permute(1, 0)

      # precompute distance/orientation (1024, 20) between grid / each panorama
      D, theta = ops.grid_helper(l_bbox, l_near_locs, self.grid_size)
      #import pdb; pdb.set_trace()
      # tile distance (num_locs, num_neighbors, 1, height, width)
      distance = D.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(  #[1024, 20, 1, 8, 32]
          (1, 1, 1, height, width))  #这里是grid的attention,要进行上采样

      # use orientation to index rays, tile (num_locs, num_neighbors, 3, height, width))
      theta_inds = torch.reshape(
          torch.round(torch.rad2deg(theta)).long(), (-1, 1))
      orientation = torch.reshape(
          self.rays[theta_inds, ...].clone().to(theta.device),
          (num_locs, num_neighbors, 3, height, width))

      # compute weights  weight.shape [1024,20,8,32], total_weight.shape:[1024,20]
      weight, total_weight = self.attention(orientation, distance, l_near_feats,
                                            l_overhead_feat)
      #import pdb; pdb.set_trace()
      # compute frobenius inner product 矩阵内积，对应元素相乘
      tmp_feats = torch.reshape(
          l_near_feats,
          [num_neighbors, num_feats, height * width]).permute(0, 2, 1) #[20, 256, 128] height=8, width = 32
      tmp_weight = torch.reshape(
          weight, [num_locs, num_neighbors, height * width]).unsqueeze(3) #[1024, 20, 256, 1]
      feats_att = torch.einsum('pic, bpid -> bpdc', tmp_feats,
                               tmp_weight).squeeze()  #[1024, 20, 128]

      # which panorama should we pay attention to?
      grid_feats = torch.sum(torch.mul(feats_att, total_weight.unsqueeze(2)),
                             dim=1)

      # form the grid
      grid_feats = grid_feats.reshape([self.grid_size, self.grid_size, -1])  #[32,32,128]
      weight = weight.reshape((                                              #[32,32,20,8,32]
          self.grid_size,
          self.grid_size,
      ) + weight.shape[1:])

      grid.append(grid_feats)
      attention.append(weight)

    return torch.stack(grid, dim=0), torch.stack(attention, dim=0)

  class GeoAttention(nn.Module):

    def __init__(self,
                 distance=False,
                 orientation=False,
                 panorama=False,
                 overhead=False):
      super().__init__()

      self.panorama = panorama
      self.overhead = overhead
      self.distance = distance
      self.orientation = orientation

      input_size = 0
      if distance:
        input_size += 1
      if orientation:
        input_size += 3
      if panorama:
        input_size += 2
      if overhead:
        input_size += 2

      # CBAM-like spatial attention: https://github.com/Jongchan/attention-module
      #                              https://salman-h-khan.github.io/papers/ICCV19-3.pdf
      kernel_sizes = [3, 5]
      self.conv1 = torch.nn.Conv2d(input_size,
                                   1,
                                   kernel_sizes[0],
                                   stride=1,
                                   padding=(kernel_sizes[0] - 1) // 2,
                                   padding_mode="circular")
      self.conv2 = torch.nn.Conv2d(input_size,
                                   1,
                                   kernel_sizes[1],
                                   stride=1,
                                   padding=(kernel_sizes[1] - 1) // 2,
                                   padding_mode="circular")
      self.finalconv = nn.Conv2d(2, 1, 1)

    def forward(self, orientation, distance, pano_feat, overhead_feat):
      """
        orientation: 1024 x 20 x 3 x 8 x 32
        distance: 1024 x 20 x 1 x 8 x 32
        pano_feat: the panorama features (e.g., 20 x 128 x 8 x 32)
        overhead_feat: the overhead feature (e.g., 1024 x 128)
      """
      #import pdb; pdb.set_trace()
      _, channel, height, width = pano_feat.shape      # channel 128, 8, 32
      num_locs, num_neighbors = orientation.shape[:2]  #1024,20

      # start with an empty tensor of the right shape
      input_ = torch.empty(num_locs,
                           num_neighbors,
                           0,
                           height,
                           width,
                           requires_grad=True).to(pano_feat.device)
      #import pdb; pdb.set_trace()
      if self.distance:
        input_ = torch.cat((input_, distance), dim=2)  #[1024, 20, 1, 8, 32]

      if self.orientation:
        input_ = torch.cat((input_, orientation), dim=2)   #[1024, 20, 4, 8, 32]

      if self.panorama:
        # process image features (max, avg), replicate to (num_locs, num_neighbors, 2, height, width)
        tmp = torch.cat(
            (torch.max(pano_feat, 1)[0].unsqueeze(1), torch.mean(
                pano_feat, 1).unsqueeze(1)),
            dim=1)
        x_pano = tmp.unsqueeze(0).repeat(num_locs, 1, 1, 1, 1)

        input_ = torch.cat((input_, x_pano), dim=2)   #[1024, 20, 6, 8, 32] num_locs=1024

      if self.overhead:
        # process image features (max, avg), replicate to (num_locs, num_neighbors, 2, height, width)
        tmp = torch.cat((torch.max(overhead_feat, 1)[0].unsqueeze(1),
                         torch.mean(overhead_feat, 1).unsqueeze(1)),
                        dim=1)
        x_overhead = tmp.unsqueeze(1).unsqueeze(3).unsqueeze(4).repeat(
            1, num_neighbors, 1, height, width)

        input_ = torch.cat((input_, x_overhead), dim=2)   #[1024, 20, 8, 8, 32]

      # reshape to (num_locs * num_neighbors, ...)
      input_ = torch.reshape(input_,
                             (num_locs * num_neighbors,) + input_.shape[2:])  #[20480, 8, 8, 32]

      # push through to get weights, reshape back to (num_locs, num_neighbors, ...)
      #import pdb; pdb.set_trace()
      c1 = self.conv1(input_)  #[20480, 1, 8, 32]
      c2 = self.conv2(input_)  #[20480, 1, 8, 32]
      fused = torch.cat((c1, c2), dim=1)    #shape [20480,2,8,32]

      weight = torch.sigmoid(
          torch.reshape(self.finalconv(fused), (
              num_locs,           #1024: 32*32
              num_neighbors,      #20
          ) + input_.shape[2:]))  #[1024, 20, 8, 32]

      total_weight = F.softmax(torch.sum(torch.flatten(weight, start_dim=2),
                                         dim=2),
                               dim=1) #[1024, 20]

      return weight, total_weight
  
      
  class GeoTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, 
                 in_channels, n_heads, d_head,
                 depth=1, dropout=0., orientation=False,distance=False,
                 panorama=False,
                 overhead=False,context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        
        self.panorama = panorama
        self.overhead = overhead
        self.distance = distance
        self.orientation = orientation
        
        input_size = 0
        if distance:
          input_size += 1
        if orientation:
          input_size += 3
        if panorama:
          input_size += 2
        if overhead:
          input_size += 2
      
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear
    
    def forward(self, orientation, distance, pano_feat, overhead_feat):
      """
        orientation: 1024 x 20 x 3 x 8 x 32
        distance: 1024 x 20 x 1 x 8 x 32
        pano_feat: the panorama features (e.g., 20 x 128 x 8 x 32)
        overhead_feat: the overhead feature (e.g., 1024 x 128)
      """
      #import pdb; pdb.set_trace()
      _, channel, height, width = pano_feat.shape      # channel 128, 8, 32
      num_locs, num_neighbors = orientation.shape[:2]  #1024,20

      # start with an empty tensor of the right shape
      input_ = torch.empty(num_locs,
                           num_neighbors,
                           0,
                           height,
                           width,
                           requires_grad=True).to(pano_feat.device)

      if self.distance:
        input_ = torch.cat((input_, distance), dim=2)  #[1024, 20, 1, 8, 32]

      if self.orientation:
        input_ = torch.cat((input_, orientation), dim=2)   #[1024, 20, 4, 8, 32]

      if self.panorama:
        # process image features (max, avg), replicate to (num_locs, num_neighbors, 2, height, width)
        tmp = torch.cat(
            (torch.max(pano_feat, 1)[0].unsqueeze(1), torch.mean(
                pano_feat, 1).unsqueeze(1)),
            dim=1)
        x_pano = tmp.unsqueeze(0).repeat(num_locs, 1, 1, 1, 1)

        input_ = torch.cat((input_, x_pano), dim=2)   #[1024, 20, 6, 8, 32] num_locs=1024

      if self.overhead:
        # process image features (max, avg), replicate to (num_locs, num_neighbors, 2, height, width)
        tmp = torch.cat((torch.max(overhead_feat, 1)[0].unsqueeze(1),
                         torch.mean(overhead_feat, 1).unsqueeze(1)),
                        dim=1)
        x_overhead = tmp.unsqueeze(1).unsqueeze(3).unsqueeze(4).repeat(
            1, num_neighbors, 1, height, width)

        input_ = torch.cat((input_, x_overhead), dim=2)   #[1024, 20, 8, 8, 32]

      # reshape to (num_locs * num_neighbors, ...)
      input_ = torch.reshape(input_,
                             (num_locs * num_neighbors,) + input_.shape[2:])  #[20480, 8, 8, 32]
      # push through to get weights, reshape back to (num_locs, num_neighbors, ...)
      import pdb; pdb.set_trace()
      x=input_
      b, c, h, w = x.shape
      x_in = x
      x = self.norm(x)
      x = self.proj_in(x)
      x = rearrange(x, 'b c h w -> b (h w) c')
      for block in self.transformer_blocks:
          x = block(x, context=context, label=label, class_ids=class_ids)
      x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
      x = self.proj_out(x)
      return x + x_in
      
      return weight, total_weight
      

class GridV2(nn.Module):

  def __init__(self, context, grid_size=32):
    super().__init__()

    self.grid_size = grid_size
    self.attention = self.GeoAttention(**context)
    self.rays = torch.tensor(ops.binned_rays(), dtype=torch.float32)

  def forward(self, bbox, near_locs, near_feats, overhead_feat):
    num_batch, num_neighbors, num_feats = near_feats.shape[:3]
    num_locs = self.grid_size**2
    height, width = self.rays.shape[2:]

    grid = []
    attention = []
    for idx in range(num_batch):
      l_bbox = bbox[idx, :]
      l_near_locs = near_locs[idx, ...]
      l_near_feats = near_feats[idx, ...]
      l_overhead_feat = None

      if overhead_feat is not None:
        l_overhead_feat = overhead_feat[idx,
                                        ...].flatten(start_dim=1).permute(1, 0)

      # precompute distance/orientation (1024, 20) between grid / each panorama
      D, theta = ops.grid_helper(l_bbox, l_near_locs, self.grid_size)

      # tile distance (num_locs, num_neighbors, 1, height, width)
      distance = D.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(
          (1, 1, 1, height, width))

      # use orientation to index rays, tile (num_locs, num_neighbors, 3, height, width))
      theta_inds = torch.reshape(
          torch.round(torch.rad2deg(theta)).long(), (-1, 1))
      orientation = torch.reshape(
          self.rays[theta_inds, ...].clone().to(theta.device),
          (num_locs, num_neighbors, 3, height, width))

      # compute weights
      import pdb; pdb.set_trace()
      weight_logits = self.attention(orientation, distance, l_near_feats,
                                     l_overhead_feat)
      weight = torch.reshape(
          F.softmax(torch.flatten(weight_logits, start_dim=1), dim=1),
          weight_logits.shape)

      # use an einsum to compute the grid_feats
      tmp_feats = l_near_feats.permute(0, 2, 3, 1)
      grid_feats = torch.einsum('pxyf,spxy->sf', [tmp_feats, weight])

      # form the grid
      grid_feats = grid_feats.reshape([self.grid_size, self.grid_size, -1])
      weight = weight.reshape((
          self.grid_size,
          self.grid_size,
      ) + weight.shape[1:])

      grid.append(grid_feats)
      attention.append(weight)

    return torch.stack(grid, dim=0), torch.stack(attention, dim=0)

  class GeoAttention(nn.Module):

    def __init__(self,
                 distance=False,
                 orientation=False,
                 panorama=False,
                 overhead=False):
      super().__init__()

      self.panorama = panorama
      self.overhead = overhead
      self.distance = distance
      self.orientation = orientation

      input_size = 0
      if distance:
        input_size += 1
      if orientation:
        input_size += 3
      if panorama:
        input_size += 2
      if overhead:
        input_size += 2

      # CBAM-like spatial attention: https://github.com/Jongchan/attention-module
      #                              https://salman-h-khan.github.io/papers/ICCV19-3.pdf
      kernel_sizes = [3, 5]
      self.conv1 = torch.nn.Conv2d(input_size,
                                   1,
                                   kernel_sizes[0],
                                   stride=1,
                                   padding=(kernel_sizes[0] - 1) // 2,
                                   padding_mode="circular")
      self.conv2 = torch.nn.Conv2d(input_size,
                                   1,
                                   kernel_sizes[1],
                                   stride=1,
                                   padding=(kernel_sizes[1] - 1) // 2,
                                   padding_mode="circular")
      self.finalconv = nn.Conv2d(2, 1, 1)

    def forward(self, orientation, distance, pano_feat, overhead_feat):
      """
        orientation: 1024 x 20 x 3 x 8 x 32
        distance: 1024 x 20 x 1 x 8 x 32
        pano_feat: the panorama features (e.g., 20 x 128 x 8 x 32)
        overhead_feat: the overhead feature (e.g., 1024 x 128)
      """
      _, channel, height, width = pano_feat.shape
      num_locs, num_neighbors = orientation.shape[:2]

      # start with an empty tensor of the right shape
      input_ = torch.empty(num_locs,
                           num_neighbors,
                           0,
                           height,
                           width,
                           requires_grad=True).to(pano_feat.device)

      if self.distance:
        input_ = torch.cat((input_, distance), dim=2)

      if self.orientation:
        input_ = torch.cat((input_, orientation), dim=2)

      if self.panorama:
        # process image features (max, avg), replicate to (num_locs, num_neighbors, 2, height, width)
        tmp = torch.cat(
            (torch.max(pano_feat, 1)[0].unsqueeze(1), torch.mean(
                pano_feat, 1).unsqueeze(1)),
            dim=1)
        x_pano = tmp.unsqueeze(0).repeat(num_locs, 1, 1, 1, 1)

        input_ = torch.cat((input_, x_pano), dim=2)

      if self.overhead:
        # process image features (max, avg), replicate to (num_locs, num_neighbors, 2, height, width)
        tmp = torch.cat((torch.max(overhead_feat, 1)[0].unsqueeze(1),
                         torch.mean(overhead_feat, 1).unsqueeze(1)),
                        dim=1)
        x_overhead = tmp.unsqueeze(1).unsqueeze(3).unsqueeze(4).repeat(
            1, num_neighbors, 1, height, width)

        input_ = torch.cat((input_, x_overhead), dim=2)
      import pdb; pdb.set_trace()
      # reshape to (num_locs * num_neighbors, ...)
      input_ = torch.reshape(input_,
                             (num_locs * num_neighbors,) + input_.shape[2:])

      # push through to get weights, reshape back to (num_locs, num_neighbors, ...)
      c1 = self.conv1(input_)
      c2 = self.conv2(input_)
      fused = torch.cat((c1, c2), dim=1)  #融合回去
      
      weight_logits = torch.reshape(self.finalconv(fused), (
          num_locs,
          num_neighbors,
      ) + input_.shape[2:])

      return weight_logits


if __name__ == "__main__":
  bbox = torch.tensor(np.random.rand(3, 4)).float()
  near_locs = torch.tensor(np.random.rand(3, 20, 2)).float()
  near_feats = torch.tensor(np.random.rand(3, 20, 128, 8, 32)).float()
  overhead_feat = torch.tensor(np.random.rand(3, 128, 32, 32)).float()

  g = Grid2d()
  grid, attention = g(bbox, near_locs, near_feats, overhead_feat)

  print(grid.shape)
  print(attention.shape)

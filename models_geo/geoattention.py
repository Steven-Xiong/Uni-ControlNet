# This reflect the attention maps of the near groud-level images

import _init_paths
import torch
import torch.nn as nn
import torch.nn.functional as F

from ops import compass_bearing, haversine_distance
from data import BrooklynQueensDataset

import imageio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def generate_rays(bearing, h_pano=16, w_pano=32):
  """Generate image where each pixel is the corresponding view ray of the pixel in cartesian coordinates."""
 
  angles = np.array(
      np.meshgrid(np.linspace(0, 2 * np.pi, w_pano), np.linspace(0, np.pi, h_pano)))
  phi = angles[0]
  theta = angles[1]
  
  x = np.sin(theta) * np.cos(phi)
  y = np.sin(theta) * np.sin(phi)
  z = np.cos(theta)
  
  img = np.stack((y, x, z), 0)
  
  #rotate by 180 degrees so that (0,1,0) is center pixel
  rot_angle = np.pi
  c = np.cos(rot_angle)
  s = np.sin(rot_angle)
  
  rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
  points = np.matmul(rotation, img.reshape((3, -1)))
  img = points.reshape(img.shape)
  
  #rotate by bearing degrees so that (0,1,0) is pointing towards target
  rot_angle = bearing
  c = np.cos(rot_angle)
  s = np.sin(rot_angle)
  
  rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
  points = np.matmul(rotation, img.reshape((3, -1)))
  img_rot = points.reshape(img.shape)

  # Note: this is trying to mimick how the panoramas were cropped 
  # adding one is a hack because x,y scale factor was slightly different
  crop_h = int(np.round(w_pano / 3.9)) + 1
  offset = int(np.round(h_pano / 4.5))
  img_crop = img_rot[:, offset:offset + crop_h - 1, :]

  return img, img_rot

def haversine_distance(lat, lon, base_lat, base_lon):
  """modified from: https://godatadriven.com/blog/the-performance-impact-of-vectorized-operations/"""

  lat_in_rad = torch.deg2rad(lat)
  lon_in_rad = torch.deg2rad(lon)
  base_lat_in_rad = torch.deg2rad(base_lat)
  base_lon_in_rad = torch.deg2rad(base_lon)

  delta_lon = lon_in_rad - base_lon_in_rad
  delta_lat = lat_in_rad - base_lat_in_rad

  inverse_angle = (torch.sin(delta_lat / 2)**2 + torch.cos(base_lat_in_rad) *
                   torch.cos(lat_in_rad) * torch.sin(delta_lon / 2)**2)
  haversine_angle = 2 * torch.asin(torch.sqrt(inverse_angle))

  EARTH_RADIUS = 6367 * 1000 # meters

  return haversine_angle * EARTH_RADIUS


def compass_bearing(lat, lon, base_lat, base_lon):
  """modified from: https://gist.github.com/jeromer/2005586"""

  lat_in_rad = torch.deg2rad(lat)
  lon_in_rad = torch.deg2rad(lon)
  base_lat_in_rad = torch.deg2rad(base_lat)
  base_lon_in_rad = torch.deg2rad(base_lon)

  delta_lon = base_lon_in_rad - lon_in_rad

  x = torch.sin(delta_lon) * torch.cos(base_lat_in_rad)
  y = torch.cos(lat_in_rad) * torch.sin(base_lat_in_rad) - (
      torch.sin(lat_in_rad) * torch.cos(base_lat_in_rad) * torch.cos(delta_lon))

  initial_bearing = torch.atan2(x, y)

  # atan2 return values from -180° to + 180°, normalize
  initial_bearing = torch.rad2deg(initial_bearing)
  compass_bearing = (initial_bearing + 360) % 360

  return torch.deg2rad(compass_bearing)


def grid_helper(bbox, locs, grid_size=32):
  num_neighbors = locs.shape[0]

  min_lat = bbox[0]
  min_lon = bbox[1]
  max_lat = bbox[2]
  max_lon = bbox[3]

  ip = torch.linspace(max_lat, min_lat, grid_size)
  jp = torch.linspace(min_lon, max_lon, grid_size)

  (II, JJ) = torch.meshgrid(ip, jp)
  II = II.to(locs.device)
  JJ = JJ.to(locs.device)

  pixel_locs = torch.cat((torch.reshape(II,
                                        (-1, 1)), torch.reshape(JJ, (-1, 1))),
                         axis=1)

  # compute distance matrix (32*32 x num_neighbors)
  D = torch.stack([
      haversine_distance(locs[idx, 0], locs[idx, 1], pixel_locs[:, 0], 
                         pixel_locs[:, 1]) for idx in range(num_neighbors)
  ],
                  dim=1)
  
  # compute orientation matrix (32*32 x num_neighbors)
  theta = torch.stack([
      compass_bearing(locs[idx, 0], locs[idx, 1], pixel_locs[:, 0], 
                          pixel_locs[:, 1]) for idx in range(num_neighbors)
  ],
                      dim=1)
  #print('D:',D/100,'D.shape:',D.shape, 'theta:',theta)
  return (D / 100), theta

def generate_rays(h_pano=16, w_pano=32):
  """Generate image where each pixel is the corresponding view ray of the pixel in cartesian coordinates."""
  angles = np.array(
    np.meshgrid(np.linspace(0, 2 * np.pi, w_pano), np.linspace(0, np.pi, h_pano)))
  phi = angles[0]
  theta = angles[1]

  x = np.sin(theta) * np.cos(phi)
  y = np.sin(theta) * np.sin(phi)
  z = np.cos(theta)

  rays = np.stack((y, x, z), 0)

  #rotate by 180 degrees so that (0,1,0) is center pixel
  rot_angle = np.pi
  c = np.cos(rot_angle)
  s = np.sin(rot_angle)

  rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
  points = np.matmul(rotation, rays.reshape((3, -1)))
  rays = points.reshape(rays.shape)

  return rays
  
def rotate_rays(rays, bearing):
  #rotate by bearing degrees so that (0,1,0) is pointing towards target
  rot_angle = bearing
  c = np.cos(rot_angle)
  s = np.sin(rot_angle)

  rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
  points = np.matmul(rotation, rays.reshape((3, -1)))
  rays_rot = points.reshape(rays.shape)

  # Note: this is trying to mimick how the panoramas were cropped 
  # adding one is a hack because x,y scale factor was slightly different
  crop_h = int(np.round(rays.shape[2] / 3.9)) + 1
  offset = int(np.round(rays.shape[1] / 4.5))
  rays_crop = rays_rot[:, offset:offset + crop_h - 1, :]

  return rays_crop
  
def binned_rays(h_pano=16, w_pano=32):
  rays = generate_rays(h_pano, w_pano)
  all_rays = []
  for angle in range(361):
      # note: this negation was necessary to make the pictures look right
      all_rays.append(rotate_rays(rays, angle*np.pi/180.))
  return np.stack(all_rays)


class GeoAttention(nn.Module):

  def __init__(self,
               distance=True,
               orientation=True,
               pano=True,
               overhead=False):
    super().__init__()

    self.pano = pano
    self.overhead = overhead
    self.distance = distance
    self.orientation = orientation

    input_size = 0
    if distance:
      input_size += 1
    if orientation:
      input_size += 3
    if pano:
      input_size += 2
    if overhead:
      input_size += 2

    # CBAM-like spatial attention: https://github.com/Jongchan/attention-module
    #                              https://salman-h-khan.github.io/papers/ICCV19-3.pdf
    kernel_sizes = [3, 5, 7]
    self.conv1 = torch.nn.Conv2d(input_size, 1, kernel_sizes[0], stride=1, padding=(kernel_sizes[0]-1) // 2, padding_mode="circular")
    self.conv2 = torch.nn.Conv2d(input_size, 1, kernel_sizes[1], stride=1, padding=(kernel_sizes[1]-1) // 2, padding_mode="circular")
    self.conv3 = torch.nn.Conv2d(input_size, 1, kernel_sizes[2], stride=1, padding=(kernel_sizes[2]-1) // 2, padding_mode="circular")
    self.finalconv = torch.nn.Conv2d(3, 1, 1)

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
    input_ = torch.empty(num_locs, num_neighbors, 0, height, width, requires_grad=True).to(pano_feat.device)
    
    if self.distance:
      input_ = torch.cat((input_, distance), dim=2)
      
    if self.orientation:
      input_ = torch.cat((input_, orientation), dim=2)

    if self.pano:
      # process image features (max, avg), replicate to (num_locs, num_neighbors, 2, height, width)
      tmp = torch.cat((torch.max(pano_feat,1)[0].unsqueeze(1), torch.mean(pano_feat,1).unsqueeze(1)), dim=1)
      x_pano = tmp.unsqueeze(0).repeat(num_locs, 1, 1, 1, 1)

      input_ = torch.cat((input_, x_pano), dim=2)

    if self.overhead:
      # process image features (max, avg), replicate to (num_locs, num_neighbors, 2, height, width)
      tmp = torch.cat((torch.max(overhead_feat,1)[0].unsqueeze(1), torch.mean(overhead_feat,1).unsqueeze(1)), dim=1)
      x_overhead = tmp.unsqueeze(1).unsqueeze(3).unsqueeze(4).repeat(1, num_neighbors, 1, height, width)
    
      input_ = torch.cat((input_, x_overhead), dim=2)
      
    # reshape to (num_locs * num_neighbors, ...)
    input_ = torch.reshape(input_, (num_locs * num_neighbors,) + input_.shape[2:])

    # push through to get weights, reshape back to (num_locs, num_neighbors, ...)
    c1 = self.conv1(input_)
    c2 = self.conv2(input_)
    c3 = self.conv3(input_)
    fused = torch.cat((c1, c2, c3), dim=1)
    weight_logits = torch.reshape(self.finalconv(fused), (num_locs, num_neighbors,) + input_.shape[2:])
    
    return weight_logits

class Grid2d(nn.Module):

  def __init__(self, grid_size=32):
    super().__init__()

    self.grid_size = grid_size
    self.attention = GeoAttention()
    self.rays = torch.tensor(binned_rays(), dtype=torch.float32)
    
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
      D, theta = grid_helper(l_bbox, l_near_locs, self.grid_size)

      # tile distance (num_locs, num_neighbors, 1, height, width)
      distance = D.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat((1, 1, 1, height, width))
      
      # use orientation to index rays, tile (num_locs, num_neighbors, 3, height, width))
      theta_inds = torch.reshape(torch.round(torch.rad2deg(theta)).long(), (-1,1))
      orientation = torch.reshape(self.rays[theta_inds, ...].clone().to(theta.device), (num_locs, num_neighbors, 3, height, width))
      
      # compute weights
      weight_logits = self.attention(orientation, distance, l_near_feats, l_overhead_feat)
      weight = torch.reshape(F.softmax(torch.flatten(weight_logits, start_dim=1), dim=1), weight_logits.shape)

      # use an einsum to compute the grid_feats
      tmp_feats = l_near_feats.permute(0,2,3,1)
      grid_feats = torch.einsum('pxyf,spxy->sf', [tmp_feats, weight])

      # form the grid
      grid_feats = grid_feats.reshape([self.grid_size, self.grid_size, -1])
      weight = weight.reshape((self.grid_size, self.grid_size,) + weight.shape[1:])

      grid.append(grid_feats)
      attention.append(weight)

    return torch.stack(grid, dim=0), torch.stack(attention, dim=0)


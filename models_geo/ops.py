import torch
import torch.nn as nn

import numpy as np


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

  EARTH_RADIUS = 6367 * 1000  # meters

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

  return (D / 100), theta


def generate_rays(h_pano=16, w_pano=32):
  """Generate image where each pixel is the corresponding view ray of the pixel in cartesian coordinates."""
  angles = np.array(
      np.meshgrid(np.linspace(0, 2 * np.pi, w_pano),
                  np.linspace(0, np.pi, h_pano)))
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


def binned_rays(h_pano=32, w_pano=64):  #原来为32， 64
  rays = generate_rays(h_pano, w_pano)
  all_rays = []
  for angle in range(361):
    all_rays.append(rotate_rays(rays, angle * np.pi / 180.))
  return np.stack(all_rays)

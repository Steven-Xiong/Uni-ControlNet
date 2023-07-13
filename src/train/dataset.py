import os
import random
import cv2
import numpy as np

from torch.utils.data import Dataset

from .util import *
from glob import glob

# add 
import io
import imageio
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import h5py
import sparse

class UniDataset(Dataset):
    def __init__(self,
                 anno_path,
                 image_dir,
                 condition_root,
                 local_type_list,
                 global_type_list,
                 resolution,
                 drop_txt_prob,
                 keep_all_cond_prob,
                 drop_all_cond_prob,
                 drop_each_cond_prob):
        
        file_ids, self.annos = read_anno(anno_path)
        self.image_paths = [os.path.join(image_dir, file_id + '.jpg') for file_id in file_ids]
        self.local_paths = {}
        for local_type in local_type_list:
            self.local_paths[local_type] = [os.path.join(condition_root, local_type, file_id + '.jpg') for file_id in file_ids]
        self.global_paths = {}
        for global_type in global_type_list:
            self.global_paths[global_type] = [os.path.join(condition_root, global_type, file_id + '.npy') for file_id in file_ids]
        
        self.local_type_list = local_type_list
        self.global_type_list = global_type_list
        self.resolution = resolution
        self.drop_txt_prob = drop_txt_prob
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.resolution, self.resolution))
        image = (image.astype(np.float32) / 127.5) - 1.0

        anno = self.annos[index]
        local_files = []
        for local_type in self.local_type_list:
            local_files.append(self.local_paths[local_type][index])
        global_files = []
        for global_type in self.global_type_list:
            global_files.append(self.global_paths[global_type][index])

        local_conditions = []
        for local_file in local_files:
            condition = cv2.imread(local_file)
            condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
            condition = cv2.resize(condition, (self.resolution, self.resolution))
            condition = condition.astype(np.float32) / 255.0
            local_conditions.append(condition)
        global_conditions = []
        for global_file in global_files:
            condition = np.load(global_file)
            global_conditions.append(condition)

        if random.random() < self.drop_txt_prob:
            anno = ''
        local_conditions = keep_and_drop(local_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        global_conditions = keep_and_drop(global_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        if len(local_conditions) != 0:
            local_conditions = np.concatenate(local_conditions, axis=2)
        if len(global_conditions) != 0:
            global_conditions = np.concatenate(global_conditions)

        return dict(jpg=image, txt=anno, local_conditions=local_conditions, global_conditions=global_conditions)
        
    def __len__(self):
        return len(self.annos)
        
def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines
  
class CVUSADataset(Dataset):
    def __init__(self,
                 anno_path,
                 image_dir,
                 condition_root,
                 local_type_list,
                 global_type_list,
                 resolution,
                 drop_txt_prob,
                 keep_all_cond_prob,
                 drop_all_cond_prob,
                 drop_each_cond_prob,
                 image_size,
                 mode):
        
        self.root = 'data/CVUSA'
        self.train_list = self.root + '/splits/train-19zl.csv'
        self.test_list = self.root + '/splits/val-19zl.csv'
        
        self.local_type_list = local_type_list
        self.global_type_list = global_type_list
        self.resolution = resolution
        self.drop_txt_prob = drop_txt_prob
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
        
        #import pdb; pdb.set_trace()
        self.mode = mode
        if self.mode == 'train':
          self.sat_files, self.street_files = self.load_path(self.train_list)
        else:
          self.sat_files, self.street_files = self.load_path(self.test_list)
        
        
    
    def load_path(self, list_filename):
        #import pdb; pdb.set_trace()
        lines = read_all_lines(list_filename)
        splits = [line.split(',') for line in lines]
        sat_images = [x[0] for x in splits]
        pano_images = [x[1] for x in splits]
        anno_images = [x[2] for x in splits]
        return sat_images, pano_images
          
    def __getitem__(self, index):
        sat_image = cv2.imread(os.path.join(self.root, self.sat_files[index]))
        sat_image = cv2.cvtColor(sat_image, cv2.COLOR_BGR2RGB)
        sat_image = cv2.resize(sat_image, (256,256))
        sat_image = sat_image.astype(np.float32) / 255.0
        #import pdb; pdb.set_trace()
        t_image = sat_image
        for i in range(1, 4):
            rot_image = np.rot90(sat_image,i)
            rot_image = np.ascontiguousarray(rot_image)
            t_image = np.hstack((t_image, rot_image))
        
        street_image = cv2.imread(os.path.join(self.root, self.street_files[index]))
        street_image = cv2.cvtColor(street_image, cv2.COLOR_BGR2RGB)
        street_image = cv2.resize(street_image, (1024,256))
        street_image = street_image.astype(np.float32) / 255.0
        
        street_seg = cv2.imread(os.path.join(self.root, self.street_files[index]).replace('panos','seg'))
        street_seg = cv2.cvtColor(street_seg, cv2.COLOR_BGR2RGB)
        street_seg = cv2.resize(street_seg, (1024,256))
        street_seg = street_seg.astype(np.float32) / 255.0
      

        local_conditions = []
        local_conditions = []
        global_conditions = []
        #for i in range(len(self.local_type_list)):
        # local contitions:
        local_conditions.append(street_seg)

        local_conditions.append(t_image)
        
        anno = 'a high resolution streetview panorama'
        # if random.random() < self.drop_txt_prob:
        #     anno = ''
        local_conditions = keep_and_drop(local_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        global_conditions = keep_and_drop(global_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        if len(local_conditions) != 0:
            local_conditions = np.concatenate(local_conditions, axis=2)
        if len(global_conditions) != 0:
            global_conditions = np.concatenate(global_conditions)

        return dict(jpg=street_image, txt=anno, local_conditions=local_conditions, global_conditions=global_conditions)
        
    def __len__(self):
        return len(self.sat_files)
      

class CVACTDataset(Dataset):
    def __init__(self,
                 anno_path,
                 image_dir,
                 condition_root,
                 local_type_list,
                 global_type_list,
                 resolution,
                 drop_txt_prob,
                 keep_all_cond_prob,
                 drop_all_cond_prob,
                 drop_each_cond_prob,
                 image_size,
                 mode):
        
        #file_ids, self.annos = read_anno(anno_path)
        #self.image_paths = [os.path.join(image_dir, file_id + '.jpg') for file_id in file_ids]
        self.local_paths = {}
        
        self.sat_dirs = []
        self.street_dirs = []
        self.street_seg_dirs = []
        self.root = 'data/CVACT'
        self.mode = mode
        if self.mode == 'train':
            self.sat_files = sorted(glob(self.root+'/ANU_data_small/satview_polish/*.jpg'))
        else:
            self.sat_files = sorted(glob(self.root+'/ANU_data_test/satview_polish/*.jpg'))
        for sat_file in self.sat_files:
            sat_dir = sat_file
            street_dir = sat_file.replace('satview_polish','streetview').replace('satView_polish','grdView')
            street_seg_dir = sat_file.replace('satview_polish','seg').replace('satView_polish','grdView')
            self.sat_dirs.append(sat_dir)
            self.street_dirs.append(street_dir)
            self.street_seg_dirs.append(street_seg_dir)
          
        #self.dataset_len = len(sat_files)
        
        # for local_type in local_type_list:
        #     self.local_paths[local_type] = [os.path.join(condition_root, local_type, file_id + '.jpg') for file_id in file_ids]
        # self.global_paths = {}
        # for global_type in global_type_list:
        #     self.global_paths[global_type] = [os.path.join(condition_root, global_type, file_id + '.npy') for file_id in file_ids]
        
        self.local_type_list = local_type_list
        self.global_type_list = global_type_list
        self.resolution = resolution
        self.drop_txt_prob = drop_txt_prob
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
    
    def __getitem__(self, index):
        sample = {}
        #import pdb; pdb.set_trace()
        # file path
        #sample_path = self.samples[index]
        
        sat_image = cv2.imread(self.sat_dirs[index])
        sat_image = cv2.cvtColor(sat_image, cv2.COLOR_BGR2RGB)
        sat_image = cv2.resize(sat_image, (256,256))
        sat_image = sat_image.astype(np.float32) / 255.0
        t_image = sat_image
        
        for i in range(1, 4):
            rot_image = np.rot90(sat_image,i)
            rot_image = np.ascontiguousarray(rot_image)
            t_image = np.hstack((t_image, rot_image))
        
        street_image = cv2.imread(self.street_dirs[index])
        street_image = cv2.cvtColor(street_image, cv2.COLOR_BGR2RGB)
        street_image = cv2.resize(street_image, (1024,256))
        street_image = street_image.astype(np.float32) / 255.0
        
        street_seg = cv2.imread(self.street_seg_dirs[index])
        street_seg = cv2.cvtColor(street_seg, cv2.COLOR_BGR2RGB)
        street_seg = cv2.resize(street_seg, (1024,256))
        street_seg = street_seg.astype(np.float32) / 255.0
        
        #anno = self.annos[index]
        
        local_conditions = []
        global_conditions = []
        #for i in range(len(self.local_type_list)):
        # local contitions:
        local_conditions.append(street_seg)

        local_conditions.append(t_image)
        
        anno = 'a high resolution streetview panorama'
        # if random.random() < self.drop_txt_prob:
        #     anno = ''
        local_conditions = keep_and_drop(local_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        global_conditions = keep_and_drop(global_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        if len(local_conditions) != 0:
            local_conditions = np.concatenate(local_conditions, axis=2)
        if len(global_conditions) != 0:
            global_conditions = np.concatenate(global_conditions)

        return dict(jpg=street_image, txt=anno, local_conditions=local_conditions, global_conditions=global_conditions)
        
    def __len__(self):
        return len(self.sat_files)
      
def ensure_dir(filename):
  if not os.path.exists(os.path.dirname(filename)):
    try:
      os.makedirs(os.path.dirname(filename))
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise


def imread(path):
  image = imageio.imread(path)
  return image


def preprocess(image):

  image = image / 255.0
  #image = np.where(image >= 0, image, 0)
  return image

# Normal distribution
def get_gauss_value(x, mu, sig):
    val = (1 / (2*np.pi*sig*sig)**0.5) * (np.exp(- (x-mu)**2 / (2*sig*sig) ) )
    return val

def make_gaussian_vector(x, y, sigma=0.75):
    inp = np.arange(start=-12, stop=13, step=1)
    x_vec = get_gauss_value(inp, mu=x, sig = sigma)
    y_vec = get_gauss_value(inp, mu=y, sig=sigma)
    return np.concatenate((x_vec, y_vec), axis=0)

def AddBorder(image, num_pixels):       
    left_border = image[:,-num_pixels:,:]
    right_border = image[:,:num_pixels,:]
    return np.hstack((left_border, image, right_border))

def AddBorder_tensor(image, num_pixels):       # tensor version
    left_border = image[:,:,-num_pixels:]
    right_border = image[:,:,:num_pixels]
    return torch.cat((left_border, image, right_border), dim=2)

class brooklynqueensdataset(Dataset):

    def __init__(self,
                 anno_path,
                 image_dir,
                 condition_root,
                 local_type_list,
                 global_type_list,
                 resolution,
                 drop_txt_prob,
                 keep_all_cond_prob,
                 drop_all_cond_prob,
                 drop_each_cond_prob,
                 image_size,
                 mode):
      #Dataset.__init__(self, opt)
      self.dirA = os.path.join(image_dir,'brooklyn/overhead/images/19') # phase 另算
      self.dirB = os.path.join(image_dir,'brooklyn/streetview/images')
      self.dirD = os.path.join(image_dir,'brooklyn/streetview/seg_256*1024')
      #self.dir_seg = os.path.join()
      name = 'brooklyn-fc8_landuse'
      neighbors=20
      #import pdb; pdb.set_trace()
      self.mode = mode #opt.mode
      self.image_size = image_size
      
      # add
      self.local_type_list = local_type_list
      self.global_type_list = global_type_list
      self.resolution = resolution
      self.drop_txt_prob = drop_txt_prob
      self.keep_all_cond_prob = keep_all_cond_prob
      self.drop_all_cond_prob = drop_all_cond_prob
      self.drop_each_cond_prob = drop_each_cond_prob
      
      if any(x in name for x in ['brooklyn', 'queens']):
      
        name, label = name.split('_')    # brooklyn-fc8  landcover
        local_name = name.split('-')[0]
        #import pdb; pdb.set_trace()
        data_dir = image_dir+'/' #"/u/eag-d1/data/near-remote/"
        self.aerial_dir = "{}{}/overhead/images/".format(data_dir, local_name)  #"{}{}/aerial/".format(data_dir, local_name)
        self.label_dir = "{}{}/labels/{}/".format(data_dir, local_name, label)
        self.streetview_dir = "{}{}/streetview/".format(data_dir, local_name)
        self.streetview_seg_dir = "{}{}/streetview/seg/".format(data_dir, local_name)
      else:
        raise ValueError('Unknown dataset.')
      
      self.name = name
      self.label = label
      self.base_dir = "data/brooklyn_queens/"  #"/u/eag-d1/scratch/scott/learn/near-remote/data/"
      self.config = self.setup(name, label, neighbors)
      mode = self.mode
      #import pdb; pdb.set_trace()
      self.h5_name = "{}_train.h5".format(name) if mode in [
          "train", "val"
      ] else "{}_test.h5".format(name)

      tmp_h5 = h5py.File("{}{}/{}".format(self.base_dir, self.name, self.h5_name),
                        'r')
      self.dataset_len = len(tmp_h5['fname'])

      if self.mode != "test":
      # use part of training for validation
        np.random.seed(1)
        inds = np.random.permutation(list(range(0, self.dataset_len)))

        K = 500
        #self.mode = 'train'
        if self.mode == "train":
          self.dataset_len = self.dataset_len - K
          self.inds = inds[:self.dataset_len]
        elif self.mode == "val":
          self.inds = inds[self.dataset_len - K:]
          self.dataset_len = K

    def setup(self, name, label, neighbors):
      config = {}
      config['loss'] = "cross_entropy"
      #import pdb; pdb.set_trace()
      # adjust output size
      if label == 'age':
        config['num_output'] = 15
        config['ignore_label'] = [0, 1]
      elif label == 'function':
        config['num_output'] = 208
        config['ignore_label'] = [0, 1]
      elif label == 'landuse':
        config['num_output'] = 13
        config['ignore_label'] = [1]
      elif label == 'landcover':
        config['num_output'] = 9
        config['ignore_label'] = [0]
      elif label == 'height':
        config['num_output'] = 2
        config['loss'] = "uncertainty"
      else:
        raise ValueError('Unknown label.')

      # setup neighbors
      config['near_size'] = neighbors

      return config

    def open_hdf5(self):
      self.h5_file = h5py.File(
        "{}{}/{}".format(self.base_dir, self.name, self.h5_name), "r")

    def open_streetview(self):
      fname = 'panos_256*1024_new.h5'   #"panos_calibrated_small.h5"
      fname_seg = 'seg_256*1024.h5'
      #import pdb; pdb.set_trace()
      self.sv_file = h5py.File("{}{}".format(self.streetview_dir, fname), "r")
      self.sv_file_seg = h5py.File("{}{}".format(self.streetview_dir, fname_seg), "r")
    def open_streetview_seg(self):
      fname = 'seg_256*1024.h5'
      self.sv_file_seg = h5py.File("{}{}".format(self.streetview_dir, fname), "r")

    def __getitem__(self,idx):
      """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
      
      #import pdb; pdb.set_trace()
      if not hasattr(self, 'h5_file'):
        self.open_hdf5()
        
      if not hasattr(self, 'sv_file'):
        self.open_streetview()
        #self.open_streetview_seg()
        
      if self.mode != "test":
        idx = self.inds[idx]
      #import pdb; pdb.set_trace()
      fname = self.h5_file['fname'][idx]     
      bbox = self.h5_file['bbox'][idx]
      label = self.h5_file['label'][idx]
      near_inds = self.h5_file['near_inds'][idx].astype(int)

      # from matlab to python indexing
      near_inds = near_inds - 1

      # setup neighbors
      if 0 < self.config['near_size'] <= near_inds.shape[-1]:  # 20 closest street-level panoramas
        near_inds = near_inds[:self.config['near_size']]
      else:
        raise ValueError('Invalid neighbor size.')

      # near locs, near feats
      sort_index = np.argsort(near_inds)            #搜索对应的最近的index
      unsort_index = np.argsort(sort_index)
      near_locs = self.h5_file['locs'][near_inds[sort_index], ...][unsort_index,
                                                                  ...]

      # decode and preprocess panoramas
      near_streetview = self.sv_file['images'][near_inds[sort_index],
                                              ...][unsort_index, ...]
      #near_streetview1 = near_streetview.astype(float)
      near_streetview_seg = self.sv_file_seg['images'][near_inds[sort_index],
                                              ...][unsort_index, ...]

      tmp = []
      for item in near_streetview:
        tmp_im = preprocess(imageio.imread(io.BytesIO(item))).transpose(
            2, 0, 1)
        tmp_im_t = torch.from_numpy(tmp_im).float()
        tmp_im_t_norm = TF.normalize(tmp_im_t,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        tmp.append(tmp_im_t)
      near_streetview = torch.stack(tmp, dim=0)
      
      tmp_seg = []
      for item in near_streetview_seg:
        tmp_im_seg = preprocess(imageio.imread(io.BytesIO(item))).transpose(
            2, 0, 1)
        tmp_im_t_seg = torch.from_numpy(tmp_im_seg).float()
        tmp_im_t_norm_seg = TF.normalize(tmp_im_t_seg,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        tmp_seg.append(tmp_im_t_seg)
      near_streetview_seg = torch.stack(tmp_seg, dim=0)

      # form absolute paths
      fname_image = "{}{}".format(self.aerial_dir, fname.decode())
      fname_label = "{}{}".format(self.label_dir, label.decode())
      fname_pano_seg = "{}{}".format(self.streetview_seg_dir, fname.decode().replace('19/',''))
      image = preprocess(imread(fname_image))  # array
      
      if self.label == "height":
        fname_label = "{}.npz".format(fname_label[:-4])
        label = sparse.load_npz(fname_label).todense()
        label = label * (1200 / 3937)  # 1 ft (US survey) = 1200/3937 m
        label = label - label.min()
      else:
        label = imread(fname_label)
  

      t_image = TF.to_tensor(image).float()
      t_label = torch.from_numpy(label).float()
      t_bbox = torch.from_numpy(bbox).float()
      t_near_locs = torch.from_numpy(near_locs).float()
      t_near_images = near_streetview
      #import pdb; pdb.set_trace()
      
      t_pano_seg = near_streetview_seg
      # paste all 4 satellite images together, transfer shape [256,256] to [256,1024]
      
      for i in range(1, 4):
            #t_image = torch.cat((t_image, transforms.ToTensor()(image).float()),2)
            #image= np.ascontiguousarray(image)
            #rot_image= image.transpose(1,0,2)[::-1]
            
            rot_image = np.rot90(image,i)
            rot_image = np.ascontiguousarray(rot_image)
            t_image = torch.cat((t_image, transforms.ToTensor()(rot_image).float()),2)
      #import pdb; pdb.set_trace()
      source_image = t_near_images[1]
      
      target_image = t_near_images[0]
      target_loc = t_near_locs[0]
      source_loc = t_near_locs[1]
      #source_loc1 = t_near_locs[2]
      
      disp_vec = np.asarray([target_loc[0]-source_loc[0], target_loc[1]-source_loc[1]])
      vec = make_gaussian_vector(disp_vec[0], disp_vec[1])
      
      ###################
      #if cfg.data_align:
      theta_x = (180.0 / np.pi) * np.arctan2(disp_vec[1], disp_vec[0])  # first y and then x i.e. arctand (y/x)

      # angle from y-axis or north
      theta_y = 90 + theta_x

      if theta_y < 0:  # fixing negative
          theta_y += 360

      column_shift = np.int(
          theta_y * (self.image_size[1]/360.0) )   

      source_image = torch.roll(source_image, column_shift, dims=2)  # rotate columns
      target_image = torch.roll(target_image, column_shift, dims=2)  # rotate columns
      source_seg = torch.roll(t_pano_seg[1], column_shift, dims=2)
      target_seg = torch.roll(t_pano_seg[0], column_shift, dims=2)
      # add near images
      near_image1 = torch.roll(t_near_images[1], column_shift, dims=2)
      near_image2 = torch.roll(t_near_images[2], column_shift, dims=2)
      
      #################
      # source_image = AddBorder_tensor(source_image, cfg.data.border_size) # border_size = 0 may led to fault
      # target_image = AddBorder_tensor(target_image, cfg.data.border_size)
      #source_image1 = source_image
    #   return {'A':t_image, 'B': t_near_images[0],'D': t_pano_seg[0], 'C': [], 'A_paths': self.aerial_dir, #self.dirA
    #            'near_locs': t_near_locs, 'near_images':t_near_images, 'label': t_label, 'bbox':t_bbox,'seg_panos':t_pano_seg,
    #            'target': target_image, 'target_loc': t_near_locs[0], 'source': source_image, 'source_loc': t_near_locs[1],
    #            'vec':vec,'source_image.shape': source_image1.shape}
    
      anno = 'a high resolution streetview panorama'
      seg_pano = target_seg #t_pano_seg[0]
      #import pdb; pdb.set_trace()
      seg_pano = seg_pano.permute(1,2,0).numpy()
      target_image = target_image.permute(1,2,0).numpy()
      t_image = t_image.permute(1,2,0).numpy()
      # 加入near images
      near_image1 = near_image1.permute(1,2,0).numpy()
      near_image2 = near_image2.permute(1,2,0).numpy()
      #t_pano_seg[0]=t_pano_seg[0].numpy().transpose(1,2,0)
      #import pdb; pdb.set_trace()
    #   seg_pano = cv2.cvtColor(np.asarray(seg_pano,dtype=np.uint8), cv2.COLOR_BGR2RGB)
    #   target_image = cv2.cvtColor(np.asarray(target_image,dtype=np.uint8), cv2.COLOR_BGR2RGB)
    #   t_image = cv2.cvtColor(np.asarray(t_image,dtype = np.uint8), cv2.COLOR_BGR2RGB)
      
      local_conditions = []
      global_conditions = []
      #for i in range(len(self.local_type_list)):
      # local contitions:
      local_conditions.append(seg_pano)

      local_conditions.append(t_image)
      
      # add 2 near images
      local_conditions.append(near_image1)
      local_conditions.append(near_image2)
      
      
      # global conditions
      #for i in range(len(self.global_type_list)):
      global_conditions.append(t_image)
      local_conditions = keep_and_drop(local_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
      global_conditions = keep_and_drop(global_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
      if len(local_conditions) != 0:
          local_conditions = np.concatenate(local_conditions, axis=2)
      if len(global_conditions) != 0:
          global_conditions = np.concatenate(global_conditions)
      return dict(jpg=target_image, txt=anno, local_conditions=local_conditions, global_conditions=global_conditions)
      #return dict(jpg=target_image, txt=anno, local_conditions=seg_pano, global_conditions=t_image)
      
  
    def __len__(self):
      return self.dataset_len


# dataset = brooklynqueensdataset()
# print(len(dataset))

# item = dataset[1234]
# jpg = item['jpg']
# txt = item['txt']
# hint = item['hint']
# print(txt)
# print(jpg.shape)
# print(hint.shape)
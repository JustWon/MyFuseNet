import os
import collections
import torch
import numpy as np
import scipy.misc as m
from torch.utils import data

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

class NYUDv2Loader_HHA(data.Dataset):
    def __init__(self, root, split="training", is_transform=False, img_size=(480, 640), img_norm=True):
        self.root = root
        self.is_transform = is_transform
        self.n_classes = 14
        self.img_norm = img_norm
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.depth_mean = 0
        self.color_files = collections.defaultdict(list)
        self.depth_files = collections.defaultdict(list)
        self.label_files = collections.defaultdict(list)
        self.cmap = self.color_map(normalized=False)

        split_map = {"training": 'train', "val": 'test',}
        self.split = split_map[split]

        for split in ["train", "test"]:
            file_list =  sorted(recursive_glob(rootdir=self.root + split +'/color/', suffix='png'))
            self.color_files[split] = file_list
        
        for split in ["train", "test"]:
            file_list =  sorted(recursive_glob(rootdir=self.root + split +'/HHA/', suffix='png'))
            self.depth_files[split] = file_list    
        
        for split in ["train", "test"]:
            file_list =  sorted(recursive_glob(rootdir=self.root + split +'/label/', suffix='png'))
            self.label_files[split] = file_list


    def __len__(self):
        return len(self.color_files[self.split])


    def __getitem__(self, index):
        color_path = self.color_files[self.split][index].rstrip()
        depth_path = self.depth_files[self.split][index].rstrip()
        label_path = self.label_files[self.split][index].rstrip()

        color_img = m.imread(color_path)    
        color_img = np.array(color_img, dtype=np.uint8)

        depth_img = m.imread(depth_path)    
        depth_img = np.array(depth_img)
        
        label_img = m.imread(label_path)    
        label_img = np.array(label_img, dtype=np.uint8)
        
        if self.is_transform:
            color_img, depth_img, label_img = self.transform(color_img, depth_img, label_img)
        
        return color_img, depth_img, label_img


    def transform(self, img, depth_img, lbl):
        img = m.imresize(img, (self.img_size[0], self.img_size[1])) # uint8 with RGB mode
        img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        
        depth_img = m.imresize(depth_img, (self.img_size[0], self.img_size[1]))
        depth_img = depth_img.astype(np.float64)
        depth_img -= self.depth_mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            depth_img = depth_img.astype(float) / 255.0
        depth_img = depth_img.transpose(2, 0, 1)
        

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        lbl = lbl[np.newaxis,:]
        lbl = lbl.astype(int)
        assert(np.all(classes == np.unique(lbl)))

        img = torch.from_numpy(img).float()
        depth_img = torch.from_numpy(depth_img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, depth_img, lbl


    def color_map(self, N=256, normalized=False):
        """
        Return Color Map in PASCAL VOC format
        """

        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap/255.0 if normalized else cmap
        return cmap


    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.cmap[l,0]
            g[temp == l] = self.cmap[l,1]
            b[temp == l] = self.cmap[l,2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb
import os
import collections
import torch
import numpy as np
from torch.utils import data
from torch.autograd import Variable
from PIL import Image

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

class NYUDv2Loader_HHA(data.Dataset):
    def __init__(self, gpu_device, root, split="training", is_transform=False, img_size=(480, 640), img_norm=True):
        self.root = root
        self.is_transform = is_transform
        self.n_classes = 13
        self.img_norm = img_norm
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.color_mean = np.array([98.185719100000,103.121196790000,121.170917550000]) # BGR
        self.hha_mean = np.array([136.302856510000, 15.202850390000, 110.708918420000 ])
        self.color_max = 255
        self.hha_max = 255
        self.color_files = collections.defaultdict(list)
        self.hha_files = collections.defaultdict(list)
        self.label_files = collections.defaultdict(list)
        self.cmap = self.color_map(normalized=False)

        split_map = {"training": 'train', "val": 'test',}
        self.split = split_map[split]

        for split in ["train", "test"]:
            file_list =  sorted(recursive_glob(rootdir=self.root + split +'/color/', suffix='png'))
            self.color_files[split] = file_list
        
        for split in ["train", "test"]:
            file_list =  sorted(recursive_glob(rootdir=self.root + split +'/HHA/', suffix='png'))
            self.hha_files[split] = file_list    
        
        for split in ["train", "test"]:
            file_list =  sorted(recursive_glob(rootdir=self.root + split +'/label/', suffix='png'))
            self.label_files[split] = file_list
        
        self.gpu_device = gpu_device
        torch.cuda.set_device(self.gpu_device)


    def __len__(self):
        return len(self.color_files[self.split])


    def __getitem__(self, index):
        color_path = self.color_files[self.split][index].rstrip()
        hha_path = self.hha_files[self.split][index].rstrip()
        label_path = self.label_files[self.split][index].rstrip()

        color_img = Image.open(color_path)    
        hha_img = Image.open(hha_path)    
        label_img = Image.open(label_path)    
        
        if self.is_transform:
            color_img, hha_img, label_img = self.transform(color_img, hha_img, label_img)
        
        return np.asarray(color_img), np.asarray(hha_img), np.asarray(label_img)


    def transform(self, color_img, hha_img, label_img):
        color_img = color_img.resize((self.img_size[1], self.img_size[0]), Image.ANTIALIAS)
        color_img = np.asarray(color_img)
        color_img = color_img[:, :, ::-1] # RGB -> BGR
        color_img = color_img.astype(np.float64)
        if self.img_norm:
            color_img -= self.color_mean
            color_img = color_img.astype(float) / self.color_max
        color_img = color_img.transpose(2, 0, 1)        # NHWC -> NCHW

        hha_img = hha_img.resize((self.img_size[1], self.img_size[0]), Image.ANTIALIAS)
        hha_img = np.asarray(hha_img)
        hha_img = hha_img.astype(np.float64)
    
        if self.img_norm:
            hha_img -= self.hha_mean  
            hha_img = hha_img.astype(float) / self.hha_max
        hha_img = hha_img.transpose(2, 0, 1)        # NHWC -> NCHW

        classes = np.unique(label_img)
        label_img = label_img.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        label_img = np.asarray(label_img)
        assert(np.all(classes == np.unique(label_img)))

        color_img = torch.from_numpy(color_img).float()
        hha_img = torch.from_numpy(hha_img).float()
        label_img = torch.from_numpy(label_img).long()

        return color_img, hha_img, label_img


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
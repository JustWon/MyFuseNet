import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class DataVisualizer:
    def __init__(self):
        self.color_mean = np.array([104.00699, 116.66877, 122.67892])
        self.depth_mean = 0
        self.color_max = 255
        self.depth_max = 5000
        self.cmap = self.color_map(normalized=False)
        self.n_classes = 14
        self.all_images = []

    def visualize_all(self, color, depth, label, result):
        self.all_images = []
        self.visualize_color_image_from_dataloader(color)
        self.visualize_depth_image_from_dataloader(depth)
        self.visualize_label_image_from_dataloader(label)
        self.visualize_label_image_from_dataloader(result)
        
        total_images = np.hstack(self.all_images)
        total_images = Image.fromarray(np.uint8(total_images))
        plt.imshow(total_images)


    def visualize_color_image_from_dataloader(self, color_img, unnormalized=True):
        color_img = np.asarray(color_img)
        color_img = np.transpose(color_img, [1,2,0])
        if unnormalized == True:
            color_img = color_img*255
            color_img += self.color_mean
            color_img = np.uint8(color_img)

        color_img = color_img[:, :, ::-1] # RGB -> BGR
        self.all_images.append(color_img)
        color_img = Image.fromarray(np.uint8(color_img))

        plt.imshow(color_img)
        
    def visualize_depth_image_from_dataloader(self, depth_img, unnormalized=True):
        depth_img = np.asarray(depth_img)
        depth_img = np.transpose(depth_img, [1,2,0])
        if unnormalized == True:
            depth_img = np.uint8(depth_img[:,:,0]*255)
        else: 
            depth_img = np.uint8(depth_img[:,:,0]/5000*255)
            
        depth_img_3ch = np.stack((depth_img,)*3, -1)
        self.all_images.append(depth_img_3ch)
        depth_img = Image.fromarray(depth_img)
        plt.imshow(depth_img)
        
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

    def decode_segmap(self, label_img):
        r = label_img.copy()
        g = label_img.copy()
        b = label_img.copy()
        for l in range(0, self.n_classes):
            r[label_img == l] = self.cmap[l,0]
            g[label_img == l] = self.cmap[l,1]
            b[label_img == l] = self.cmap[l,2]

        rgb = np.zeros((label_img.shape[0], label_img.shape[1], 3))
        rgb[:, :, 0] = r 
        rgb[:, :, 1] = g 
        rgb[:, :, 2] = b 
        return rgb

    def visualize_label_image_from_dataloader(self, label_img):
        label_img = np.asarray(label_img)
        color_labeld_img = self.decode_segmap(label_img)
        color_labeld_img = np.uint8(color_labeld_img)
        self.all_images.append(color_labeld_img)
        color_labeld_img = Image.fromarray(color_labeld_img)
        plt.imshow(color_labeld_img)
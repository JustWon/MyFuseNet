import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visualize_color_image_from_dataloader(color_img, unnormalized=False):
    color_img = np.transpose(color_img, [1,2,0])
    color_img = color_img[:, :, ::-1] # RGB -> BGR
    if unnormalized == True:
        color_img = Image.fromarray(np.uint8(color_img*255))
    else:
        color_img = Image.fromarray(np.uint8(color_img))
    plt.imshow(color_img)
    
    
    
def visualize_depth_image_from_dataloader(depth_img, unnormalized=False):
    depth_img = np.transpose(depth_img, [1,2,0])
    if unnormalized == True:
        depth_img = Image.fromarray(np.uint8(depth_img[:,:,0]*255))
    else: 
        depth_img = Image.fromarray(np.uint8(depth_img[:,:,0]/5000*255))
    plt.imshow(depth_img)
    
def color_map(N=256, normalized=False):
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

n_classes = 14
cmap = color_map()

def decode_segmap(label_img):
    r = label_img.copy()
    g = label_img.copy()
    b = label_img.copy()
    for l in range(0, n_classes):
        r[label_img == l] = cmap[l,0]
        g[label_img == l] = cmap[l,1]
        b[label_img == l] = cmap[l,2]

    rgb = np.zeros((label_img.shape[0], label_img.shape[1], 3))
    rgb[:, :, 0] = r 
    rgb[:, :, 1] = g 
    rgb[:, :, 2] = b 
    return rgb

def visualize_label_image_from_dataloader(label_img):
    # label_img = np.transpose(label_img, [1,2,0])
    label_img = np.asarray(label_img)
    color_labeld_img = decode_segmap(label_img)
    color_labeld_img = Image.fromarray(np.uint8(color_labeld_img))
    plt.imshow(color_labeld_img)
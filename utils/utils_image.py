import os
from typing import TYPE_CHECKING, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:
    import numpy.typing as npt
'''
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# 03/Mar/2019
# --------------------------------------------
# https://github.com/twhui/SRGAN-pyTorch
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp',
    '.BMP', '.tif'
]


def is_image_file(filename: str):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


'''
# --------------------------------------------
# get image pathes
# --------------------------------------------
'''


def get_image_paths(dataroot: str) -> List[str]:
    paths = None  # return None if dataroot is None
    if dataroot is not None:
        paths = sorted(_get_paths_from_images(dataroot))
    return paths


def _get_paths_from_images(path: str) -> List[str]:
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images: List[str] = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


'''
# --------------------------------------------
# makedir
# --------------------------------------------
'''


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


'''
# --------------------------------------------
# read image from path
# opencv is fast, but read BGR numpy image
# --------------------------------------------
'''


# --------------------------------------------
# get uint8 image of size HxWxn_channles (RGB)
# --------------------------------------------
def imread_uint(path: str, n_channels: int = 3) -> 'npt.NDArray[np.float64]':
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


# --------------------------------------------
# matlab's imwrite
# --------------------------------------------
def imsave(img: 'npt.NDArray[np.float64]', img_path: str):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


'''
# --------------------------------------------
# image format conversion
# --------------------------------------------
# numpy(single) <--->  numpy(unit)
# numpy(single) <--->  tensor
# numpy(unit)   <--->  tensor
# --------------------------------------------
'''

# --------------------------------------------
# numpy(single) [0, 1] <--->  numpy(unit)
# --------------------------------------------


def uint2single(img):

    return np.float32(img / 255.)


# --------------------------------------------
# numpy(unit) (HxWxC or HxW) <--->  tensor
# --------------------------------------------


# convert uint to 3-dimensional torch tensor
def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(
        2, 0, 1).float().div(255.)


# convert 2/3/4-dimensional torch tensor to uint
def tensor2uint(tensor):
    img = tensor.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


# --------------------------------------------
# numpy(single) (HxWxC) <--->  tensor
# --------------------------------------------


# convert single (HxWxC) to 3-dimensional torch tensor
def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


def save_d(d: torch.Tensor, path: str):
    def merge_images(image_batch: np.ndarray):
        """
            d: C_out, C_in, d_size, d_size
        """
        h, w = image_batch.shape[-2], image_batch.shape[-1]
        img = np.zeros((int(h * 8 + 7), int(w * 8 + 7)))
        for idx, im in enumerate(image_batch):
            i = idx % 8 * (h + 1)
            j = idx // 8 * (w + 1)

            img[j:j + h, i:i + w] = im
        img = cv2.resize(img,
                         dsize=(256, 256),
                         interpolation=cv2.INTER_NEAREST)
        return img

    d = d.mean(0).numpy()
    d = np.where(d > np.quantile(d, 0.75), 0, d)
    d = np.where(d < np.quantile(d, 0.25), 0, d)

    im_merged = merge_images(d)
    im_merged = np.absolute(im_merged)
    plt.imsave(path,
               im_merged,
               cmap='Greys',
               vmin=im_merged.min(),
               vmax=im_merged.max())

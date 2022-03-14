import random
from typing import Any, Dict

import numpy as np
import torch
import utils.utils_image as util
from scipy import ndimage
from scipy.io import loadmat
from numpy import float32
from glob import glob
from .dataset_ir import DatasetIR


class DatasetSR(DatasetIR):
    def __init__(self, opt_dataset: Dict[str, Any]):
        super().__init__(opt_dataset)
        self.kernel_size = opt_dataset['k_size']
        self.tag = f"{self.sigma}"

        self.kernel_paths = glob(f"kernels/{opt_dataset['phase']}/*/*.mat")
        self.sf = opt_dataset['sf']

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # -------------------------------------
        # get H image
        # -------------------------------------
        img_index = index

        H_path = self.paths_H[img_index]
        img_H = util.imread_uint(H_path, self.n_channels)

        img_H = img_H[..., :int(img_H.shape[-3] // self.sf *
                                self.sf), :int(img_H.shape[-2] // self.sf *
                                               self.sf), :]
        # kernel
        kernel_index = random.choice(range(len(self.kernel_paths)))
        kernel_path = self.kernel_paths[kernel_index]
        k = loadmat(kernel_path)['kernel']

        # noise level
        noise_level = self.sigma / 255.0

        # generate image
        img_L = ndimage.filters.correlate(img_H,
                                          np.expand_dims(k, axis=2),
                                          mode='wrap')
        img_L = img_L[::self.sf, ::self.sf, ...]
        img_L = util.uint2single(img_L) + np.random.normal(
            0, noise_level, img_L.shape)

        k = util.single2tensor3(np.expand_dims(float32(k),
                                               axis=2)).unsqueeze(0)
        img_H, img_L = util.uint2tensor3(img_H), util.single2tensor3(img_L)
        noise_level = torch.FloatTensor([noise_level
                                         ]).unsqueeze(1).unsqueeze(1)

        self.count += 1
        return {
            'y': img_L,
            'y_gt': img_H,
            'k_gt': k,
            'sigma': noise_level,
            'sf': self.sf,
            'path': H_path
        }

    def __len__(self):
        return len(self.paths_H)
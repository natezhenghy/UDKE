from typing import Any, Dict, Union, List
import torch.utils.data as data
import utils.utils_image as util
import os


class DatasetIR(data.Dataset):
    def __init__(self, opt_dataset: Dict[str, Any]) -> None:
        super().__init__()
        self.opt = opt_dataset
        self.n_channels: int = opt_dataset['n_channels']
        self.patch_size: int = self.opt['H_size']

        self.sigma: Union[List[int], int] = opt_dataset['sigma']

        self.paths_H = util.get_image_paths(opt_dataset['dataroot_H'])
        self.count: int = 0
        self.name: str = os.path.basename(opt_dataset['dataroot_H'])
        self.tag: str = ""

    def __len__(self):

        return len(self.paths_H)
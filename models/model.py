import os
from typing import Any, Dict

import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.nn.parallel import DataParallel

from models.select_network import init_net
from utils import utils_image as util


class Model:
    def __init__(self, opt: Dict[str, Any]):
        self.opt = opt
        self.save_dir = opt['path']['models']
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')

        self.netG = init_net(opt).to(self.device)
        self.netG = DataParallel(self.netG)
        self.netG.eval()

        self.metrics: Dict[str, Any] = {'psnr': 0, 'ssim': 0, 'psnr_k': 0}

    def init(self):
        self.opt_test = self.opt['test']
        self.load_model()

    def feed_data(self, data: Dict[str, Any]):
        self.y = data['y'].to(self.device)
        self.y_gt = data['y_gt'].to(self.device)
        self.k_gt = data['k_gt'].to(self.device)
        self.sigma = data['sigma'].to(self.device)
        self.sf = data['sf'][0].item()
        self.path = data['path']

    def test(self):
        with torch.no_grad():
            self.dx, self.k, self.d = self.netG(self.y, self.sigma, self.sf)
        self.prepare_visuals()

    def prepare_visuals(self):
        self.out_dict: Dict[str, Any] = {}
        self.out_dict['y'] = util.tensor2uint(self.y[0].detach().float().cpu())
        self.out_dict['dx'] = util.tensor2uint(
            self.dx[0].detach().float().cpu())
        self.out_dict['y_gt'] = util.tensor2uint(
            self.y_gt[0].detach().float().cpu())
        self.out_dict['path'] = self.path[0]
        self.out_dict['k'] = util.tensor2uint(self.k[0].detach().float().cpu())
        self.out_dict['k_gt'] = util.tensor2uint(
            self.k_gt[0].detach().float().cpu())

    def cal_metrics(self):
        self.metrics['psnr'] = peak_signal_noise_ratio(self.out_dict['dx'],
                                                       self.out_dict['y_gt'])
        self.metrics['ssim'] = structural_similarity(self.out_dict['dx'],
                                                     self.out_dict['y_gt'],
                                                     multichannel=True)
        self.metrics['psnr_k'] = peak_signal_noise_ratio(
            self.out_dict['k'], self.out_dict['k_gt'])

        return self.metrics['psnr'], self.metrics['ssim'], self.metrics[
            'psnr_k']

    def save_visuals(self, tag: str):
        y_img = self.out_dict['y']
        y_gt_img = self.out_dict['y_gt']
        dx_img = self.out_dict['dx']
        path = self.out_dict['path']

        img_name = os.path.splitext(os.path.basename(path))[0]
        img_dir = os.path.join(self.opt['path']['images'], img_name)
        os.makedirs(img_dir, exist_ok=True)

        save_img_path = os.path.join(img_dir, f"{img_name:s}_{tag}.png")
        util.imsave(dx_img, save_img_path)
        util.imsave(y_img, save_img_path.replace('.png', '_y.png'))
        util.imsave(y_gt_img, save_img_path.replace('.png', '_y_gt.png'))
        util.imsave(self.out_dict['k_gt'],
                    save_img_path.replace('.png', '_k_gt.png'))
        util.imsave(self.out_dict['k'],
                    save_img_path.replace('.png', '_k.png'))

    def load_model(self):
        load_path = os.path.join(self.opt['path']['root'],
                                 self.opt['path']['pretrained_netG'])

        print(f'Loading model from {load_path}')

        network = self.netG

        if isinstance(network, nn.DataParallel):
            network = network.module

        # load head
        network.head.load_state_dict(  # type: ignore
            torch.load(os.path.join(load_path, 'models', 'head.pth')),
            strict=True)

        # load x
        state_dict_x = torch.load(os.path.join(load_path, 'models', 'x.pth'))
        network.body.net_x.load_state_dict(  # type: ignore
            state_dict_x, strict=False)

        # load k
        path_k = os.path.join(load_path, 'models', 'k.pth')
        if os.path.exists(path_k) and network.body.net_k is not None:
            print('load kernel net')
            network.body.net_k.load_state_dict(  # type: ignore
                torch.load(path_k), strict=True)

        # load hypa
        state_dict_hypa = torch.load(
            os.path.join(load_path, 'models', 'hypa.pth'))
        network.hypa_list.load_state_dict(  # type: ignore
            state_dict_hypa, strict=True)

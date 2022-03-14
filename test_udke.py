import argparse
import faulthandler
import logging
import os
import os.path
import random
from typing import List

import numpy as np
import torch
from prettytable import PrettyTable
from torch import cuda
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_ir import DatasetIR
from data.select_dataset import define_Dataset
from models.model import Model
from utils import utils_image as util
from utils import utils_logger
from utils import utils_option as option

faulthandler.enable()
torch.autograd.set_detect_anomaly(True)


def main(json_path: str = 'options/test_udke.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt',
                        type=str,
                        default=json_path,
                        help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.mkdirs(
        (path for key, path in opt['path'].items() if 'pretrained' not in key))

    option.save(opt)

    opt: option.NoneDict = option.dict_to_nonedict(opt)  # type: ignore

    # logger
    logger_name = 'train'
    utils_logger.logger_info(
        logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    seed = random.randint(1, 10000)

    # data
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)

    opt_data_test = opt["data"]["test"]
    test_sets: List[DatasetIR] = define_Dataset(opt_data_test)
    test_loaders: List[DataLoader] = []
    for test_set in test_sets:
        test_loaders.append(
            DataLoader(test_set,
                       batch_size=1,
                       shuffle=False,
                       num_workers=1,
                       drop_last=True,
                       pin_memory=True))

    # model
    model = Model(opt)
    model.init()

    avg_psnrs = {}
    avg_ssims = {}
    avg_psnrs_k = {}
    tags = []
    test_index = 0
    for test_loader in tqdm(test_loaders):
        test_set: DatasetIR = test_loader.dataset  # type: ignore
        avg_psnr = 0.
        avg_ssim = 0.
        avg_psnr_k = 0.
        for test_data in tqdm(test_loader):
            test_index += 1
            model.feed_data(test_data)
            model.test()
            psnr, ssim, psnr_k = model.cal_metrics()
            avg_psnr += psnr
            avg_ssim += ssim
            avg_psnr_k += psnr_k

            model.save_visuals(test_set.tag)
        avg_psnr = round(avg_psnr / len(test_loader), 2)
        avg_ssim = round(avg_ssim * 100 / len(test_loader), 2)
        avg_psnr_k = round(avg_psnr_k / len(test_loader), 2)

        name = test_set.name

        if name in avg_psnrs:
            avg_psnrs[name].append(avg_psnr)
            avg_ssims[name].append(avg_ssim)
            avg_psnrs_k[name].append(avg_psnr_k)
        else:
            avg_psnrs[name] = [avg_psnr]
            avg_ssims[name] = [avg_ssim]
            avg_psnrs_k[name] = [avg_psnr_k]
        if test_set.tag not in tags:
            tags.append(test_set.tag)
    header = ['Dataset'] + tags
    t = PrettyTable(header)
    for key, value in avg_psnrs.items():
        t.add_row([key] + value)
    logger.info(f"Test PSNR:\n{t}")

    t = PrettyTable(header)
    for key, value in avg_ssims.items():
        t.add_row([key] + value)
    logger.info(f"Test SSIM:\n{t}")

    t = PrettyTable(header)
    for key, value in avg_psnrs_k.items():
        t.add_row([key] + value)
    logger.info(f"Test Kernel PSNR:\n{t}")


if __name__ == '__main__':
    main()

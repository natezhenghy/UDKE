'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# --------------------------------------------
'''
from glob import glob
import os
from data.dataset_sr import DatasetSR
from copy import deepcopy


def define_Dataset(opt_dataset):
    datasets = []
    paths = glob(os.path.join(opt_dataset['dataroot_H'], '*'))
    sigmas = opt_dataset['sigma']
    sfs = opt_dataset['sf']
    opt_dataset_sub = deepcopy(opt_dataset)
    for path in paths:
        for sigma in sigmas:
            for sf in sfs:
                opt_dataset_sub['dataroot_H'] = path
                opt_dataset_sub['sigma'] = sigma
                opt_dataset_sub['sf'] = sf
                datasets.append(DatasetSR(opt_dataset_sub))
    return datasets

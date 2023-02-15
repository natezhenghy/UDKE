# Unfolded Deep Kernel Estimation for Blind Image Super-resolution
Hongyi Zheng, Hongwei Yong, Lei Zhang, "Unfolded Deep Kernel Estimation for Blind Image Super-resolution". Accepted by ECCV2022.

[[paper]](https://www4.comp.polyu.edu.hk/~cslzhang/paper/ECCV2022_UDKE.pdf) [[supp]](https://www4.comp.polyu.edu.hk/~cslzhang/paper/ECCV2022_UDKE_supp.pdf)

The implementation of UDKE is based on the awesome Image Restoration Toolbox [[KAIR]](https://github.com/cszn/KAIR).

## Requirement
- PyTorch 1.9+
- prettytable
- tqdm

## Testing
**Step 1**

- Download testing kernels from [[OneDrive]](https://1drv.ms/u/s!ApI9l49EgrUbkesl_RplE66v51o7Wg?e=IrSexF).
- Unzip downloaded testing kernels and put the folders into ```./kernels/test```
- Download pretrained models from [[OneDrive]](https://1drv.ms/u/s!ApI9l49EgrUbkesl_RplE66v51o7Wg?e=IrSexF).
- Unzip downloaded file and put the folders into ```./release/udke```

**Step 2**

Configure ```options/test_udke.json```. Important settings:
- task: task name.
- path/root: path to save the tasks.
- path/pretrained_netG: path to the folder containing the pretrained models.
- data/test/sigma: noise level
- data/test/sf: scale factor
- data/test/dataroot_h: path to testing sets

**Step 3**
```bash
python test_udke.py
```
## Training

**Step 1**

Configure ```options/train_udke.json```. Important settings:
- task: task name.
- path/root: path to save the tasks.
- data/train/sigma: noise level
- data/train/sf: scale factor
- data/train/dataroot_h: path to traing sets
- data/test/sigma: noise level
- data/test/sf: scale factor
- data/test/dataroot_h: path to testing sets

**Step 2**
```bash
python train_udke.py
```

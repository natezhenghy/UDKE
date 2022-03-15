# Unfolded Deep Kernel Estimation for Blind Image Super-resolution
Hongyi Zheng, Hongwei Yong, Lei Zhang, "Unfolded Deep Kernel Estimation for Blind Image Super-resolution".

[[arxiv]](https://arxiv.org/abs/2203.05568)

The implementation of UDKE is based on the awesome Image Restoration Toolbox [[KAIR]](https://github.com/cszn/KAIR).

## Requirement
- PyTorch 1.9+
- prettytable
- tqdm

## Testing
**Step 1**

- Download testing kernels from [[OneDrive]](......) or [[BaiduPan]](......) (password: ****).
- Unzip downloaded testing kernels and put the folders into ```./kernels/test```
- Download pretrained models from [[OneDrive]](......) or [[BaiduPan]](......) (password: ****).
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

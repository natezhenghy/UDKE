from math import ceil
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

import models.basicblock as B

from .utils import *


class HeadNet(nn.Module):
    def __init__(self, k_size: int):
        super(HeadNet, self).__init__()

        self.head_k = torch.zeros(1, 1, 1, k_size, k_size)

    def forward(
            self,
            y: torch.Tensor,
            sf: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.upsample(y, scale_factor=sf, mode='bicubic')
        d = None
        k = self.head_k.repeat(x.size(0), 1, 1, 1, 1).to(x.device)

        return x, d, k


class BodyNet(nn.Module):
    def __init__(self, in_nc: int, nc_x: List[int], nc_d: List[int],
                 nc_k: List[int], out_nc: int, nb: int,
                 multi_stage: bool) -> None:
        super(BodyNet, self).__init__()

        self.net_x: nn.Module = NetX(in_nc=in_nc,
                                     nc_x=nc_x,
                                     out_nc=out_nc,
                                     nb=nb)
        self.solve_fft = SolveFFT()

        self.net_d: Optional[nn.Module] = None

        self.net_k: Optional[nn.Module] = None
        self.net_k = NetK(nc_k=nc_k)

        self.solve_ls = SolveLS()

        self.multi_stage = multi_stage

    def normalize_k(self, k: torch.Tensor) -> torch.Tensor:
        """
            k: N, 1, 1, k_size, k_size
        """
        return k / k.sum(dim=[1, 2, 3, 4]).unsqueeze(-1).unsqueeze(
            -1).unsqueeze(-1).unsqueeze(-1) + 1e-10

    def forward(self,
                k: torch.Tensor,
                x: torch.Tensor,
                d: torch.Tensor,
                y: torch.Tensor,
                y_: torch.Tensor,
                alpha_k: torch.Tensor,
                beta_k: torch.Tensor,
                alpha_x: torch.Tensor,
                beta_x: torch.Tensor,
                stage: int,
                sf: int = 1) -> torch.Tensor:
        """
            x: N, C_in, H, W or N, C_out, H, W
            d: N, C_out, C_in, d_size, d_size
            k: N, 1, 1, k_size, k_size
            y_: N, C_out, 1, H, W, 2
            y: N, C_out, H, W
            alpha/beta: 1, 1, 1, 1
            reg: float
        """
        onesided = sf == 1

        # x
        # solve x
        if stage != 0 or True:
            x_ = rfft(x, onesided=onesided)
            k_ = p2o(k, x.shape[-2:], onesided=onesided)
            x = self.solve_fft(x_.unsqueeze(2), k_, y_, alpha_x, sf=sf)
        # net x
        beta_x = (1 / beta_x.sqrt()).repeat(1, 1, x.size(2), x.size(3))
        x = self.net_x(torch.cat([x, beta_x], dim=1))

        # k
        if self.multi_stage:
            # solve k
            dx = x

            k = self.solve_ls(dx.unsqueeze(2),
                              k,
                              y.unsqueeze(2),
                              alpha_k,
                              sf=sf,
                              stage=stage)
            k = self.normalize_k(k)

            # net k
            beta_k = (1 / beta_k.sqrt()).repeat(1, 1, k.size(3), k.size(4))
            size_k = [k.size(1), k.size(2)]
            k = k.view(k.size(0), k.size(1) * k.size(2), k.size(3), k.size(4))
            k = self.net_k(torch.cat([k, beta_k], dim=1))
            k = k.view(k.size(0), size_k[0], size_k[1], k.size(2), k.size(3))

        return k, x, d


class NetX(nn.Module):
    def __init__(self,
                 in_nc: int = 65,
                 nc_x: List[int] = [64, 128, 256, 512],
                 out_nc: int = 64,
                 nb: int = 2):
        super(NetX, self).__init__()

        self.m_head = B.conv(in_nc, nc_x[0], bias=False, mode='C')
        in_nc = nc_x[0]

        self.m_down1 = B.sequential(
            *[
                B.ResBlock(in_nc, in_nc, bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(in_nc, nc_x[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(
            *[
                B.ResBlock(nc_x[1], nc_x[1], bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(nc_x[1], nc_x[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(
            *[
                B.ResBlock(nc_x[2], nc_x[2], bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(nc_x[2], nc_x[3], bias=False, mode='2'))

        self.m_body = B.sequential(*[
            B.ResBlock(nc_x[-1], nc_x[-1], bias=False, mode='CRC')
            for _ in range(nb)
        ])

        self.m_up3 = B.sequential(
            B.upsample_convtranspose(nc_x[3], nc_x[2], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[2], nc_x[2], bias=False, mode='CRC')
                for _ in range(nb)
            ])
        self.m_up2 = B.sequential(
            B.upsample_convtranspose(nc_x[2], nc_x[1], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[1], nc_x[1], bias=False, mode='CRC')
                for _ in range(nb)
            ])
        self.m_up1 = B.sequential(
            B.upsample_convtranspose(nc_x[1], nc_x[0], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[0], nc_x[0], bias=False, mode='CRC')
                for _ in range(nb)
            ])

        self.m_tail = B.conv(nc_x[0], out_nc, bias=False, mode='C')

    def forward(self, x: torch.Tensor):
        # padding
        h, w = x.size()[-2:]
        paddingBottom = int(ceil(h / 8) * 8 - h)
        paddingRight = int(ceil(w / 8) * 8 - w)
        x = F.pad(x, [0, paddingRight, 0, paddingBottom], mode='circular')

        x = self.m_head(x)
        x1 = x
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        x = x[..., :h, :w]
        return x


class SolveFFT(nn.Module):
    def __init__(self):
        super(SolveFFT, self).__init__()

    def forward(self,
                x_: torch.Tensor,
                d_: torch.Tensor,
                y_: torch.Tensor,
                alpha: torch.Tensor,
                HW: Optional['npt.NDArray[np.float64]'] = None,
                sf: int = 1) -> torch.Tensor:
        """
            x_: agg - N, 1, C_in, H, W
                not agg - N, C_out, 1, H, W
            d_: agg - N, C_out, C_in, H, W
                not agg - N, 1, 1, H, W
            y_: N, C_out, 1, H, W
            alpha: N, 1, 1, 1
        """

        onesided = sf == 1

        # alpha: N, 1, 1, 1, 1
        alpha = alpha.unsqueeze(-1)

        _d = torch.conj(d_)
        _dd_ = _d * d_
        z_ = y_ * d_ + alpha * x_
        _dz_ = _d * z_
        if x_.shape[2] > 1:
            _dd_ = _dd_.sum(2, keepdim=True)
            _dz_ = _dz_.sum(2, keepdim=True)
        if sf > 1:
            _dz_ = torch.mean(splits(_dz_.squeeze(2), sf),
                              dim=-1,
                              keepdim=False).unsqueeze(2)
            _dd_ = torch.mean(splits(_dd_.squeeze(2), sf),
                              dim=-1,
                              keepdim=False).unsqueeze(2)
        _dd_a = _dd_ + alpha
        q_ = _dz_ / _dd_a

        if sf > 1:
            q_ = q_.repeat(1, 1, 1, sf, sf)
        d_q_ = d_ * q_
        x_ = (z_ - d_q_) / alpha
        if x_.shape[2] > 1:
            x_ = x_.mean(1)
        else:
            x_ = x_.squeeze(2)
        if onesided:
            x = irfft(x_, onesided=True, s=list(HW))
        else:
            x = irfft(x_, onesided=False)
        return x


class NetD(nn.Module):
    def __init__(self,
                 nc_d: List[int] = [16],
                 out_nc: int = 1,
                 expand: int = 1):
        super(NetD, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(out_nc * nc_d[0] + 1,
                      out_nc * nc_d[0] * expand,
                      3,
                      padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_nc * nc_d[0] * expand,
                      out_nc * nc_d[0],
                      3,
                      padding=1))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(out_nc * nc_d[0],
                      out_nc * nc_d[0] * expand,
                      3,
                      padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_nc * nc_d[0] * expand,
                      out_nc * nc_d[0],
                      3,
                      padding=1))
        self.mlp3 = nn.Sequential(
            nn.Conv2d(out_nc * nc_d[0],
                      out_nc * nc_d[0] * expand,
                      3,
                      padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_nc * nc_d[0] * expand,
                      out_nc * nc_d[0],
                      3,
                      padding=1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, d):
        d1 = d
        d = self.relu(self.mlp(d))
        d = self.relu(self.mlp2(d))
        d = self.mlp3(d) + d1[:, :-1, :, :]
        return d


class NetK(nn.Module):
    def __init__(self, nc_k: List[int] = [16], expand: int = 1):
        super(NetK, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(1 + 1, nc_k[0] * expand, 3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(nc_k[0] * expand, nc_k[0], 3, padding=1))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(nc_k[0], nc_k[0] * expand, 3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(nc_k[0] * expand, nc_k[0], 3, padding=1))
        self.mlp3 = nn.Sequential(
            nn.Conv2d(nc_k[0], nc_k[0] * expand, 3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(nc_k[0] * expand, 1, 3, padding=1))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        h, w = k.size()[-2:]
        paddingBottom = int(ceil(h / 2) * 2 - h)
        paddingRight = int(ceil(w / 2) * 2 - w)
        k = F.pad(k, [0, paddingRight, 0, paddingBottom], mode='circular')

        k1 = k
        k = self.relu(self.mlp(k))
        k = self.relu(self.mlp2(k))
        k = self.mlp3(k)
        k += k1[:, 0, :, :].unsqueeze(1)
        k = self.relu(k)
        k = k[..., :h, :w]
        return k


class LUSolve(autograd.Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, b: torch.Tensor):
        l, u = torch.lu(A)
        d = torch.lu_solve(b, l, u)  # D = Q-1 @ P
        ctx.save_for_backward(l, u, d)
        return d

    @staticmethod
    def backward(ctx, dldd: torch.Tensor):
        l, u, d = ctx.saved_tensors
        dldp = torch.lu_solve(dldd, l, u)
        dldq = -dldp.matmul(d.transpose(-2, -1))

        return dldq, dldp


class CholeskySolve(autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, p: torch.Tensor):
        l = torch.cholesky(q)
        d = torch.cholesky_solve(p, l)  # D = Q-1 @ P
        ctx.save_for_backward(l, d)
        return d

    @staticmethod
    def backward(ctx, dldd: torch.Tensor):
        l, d = ctx.saved_tensors
        dldp = torch.cholesky_solve(dldd, l)
        dldq = -dldp.matmul(d.transpose(-2, -1))

        return dldq, dldp


class LSTSQSolveUniversal(autograd.Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, b: torch.Tensor, N: torch.Tensor):
        d = torch.linalg.lstsq(A, b, driver='gelsd')[0]
        ctx.save_for_backward(A, d, N)
        return d

    @staticmethod
    def backward(ctx, dldd: torch.Tensor):
        A, d, N = ctx.saved_tensors
        dldp = torch.linalg.lstsq(A,
                                  dldd.repeat(N.item(), 1, 1).view(1, -1, 1),
                                  driver='gelsd')[0]
        dldq = -dldp.matmul(d.transpose(-2, -1))
        dldq, dldp = dldq.repeat(N.item(), 1,
                                 1).view(1, -1, dldq.shape[2]), dldp.repeat(
                                     N.item(), 1,
                                     dldp.shape[2]).view(1, -1, 1)
        return dldq, dldp, N


class LSTSQSolve(autograd.Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, b: torch.Tensor):
        d = torch.linalg.lstsq(A, b, driver='gelsd')[0]
        ctx.save_for_backward(A, d)
        return d

    @staticmethod
    def backward(ctx, dldd: torch.Tensor):
        A, d = ctx.saved_tensors
        dldp = torch.linalg.lstsq(A, dldd, driver='gelsd')[0]
        dldq = -dldp.matmul(d.transpose(-2, -1))

        return dldq, dldp


class SolveLS(nn.Module):
    def __init__(self, optimize_memory: bool = False):
        super(SolveLS, self).__init__()

        self.cholesky_solve: Callable[[torch.Tensor, torch.Tensor],
                                      torch.Tensor] = CholeskySolve.apply
        self.lu_solve: Callable[[torch.Tensor, torch.Tensor],
                                torch.Tensor] = LUSolve.apply
        self.lstsq_solve: Callable[[torch.Tensor, torch.Tensor],
                                   torch.Tensor] = LSTSQSolve.apply
        self.lstsqu_solve: Callable[[torch.Tensor, torch.Tensor],
                                    torch.Tensor] = LSTSQSolveUniversal.apply
        self.optimize_memory = optimize_memory
        self.mosaicing = False

    def forward(self,
                x: torch.Tensor,
                d: torch.Tensor,
                y: torch.Tensor,
                alpha: torch.Tensor,
                sf: int = 1,
                stage: int = 0):
        """
            x (dictionary): N, 1, C_in, H, W
            x (kernel): N, C_out, 1, H, W
            d: N, C_out, C_in, D, D
            y: N, C_out, 1, H, W

            alpha: N, 1, 1, 1
            reg: float
        """
        N, C_out = x.shape[:2]

        D = d.shape[-1]

        offset = 1 if sf in [2, 3] else 3
        # N, C_out * D ** 2, H_crop * W_crop
        x_offset = x.squeeze(2)[..., offset:, offset:]
        H_offset, W_offset = x_offset.shape[-2:]
        H_crop = (H_offset - D) // sf + 1
        W_crop = (W_offset - D) // sf + 1
        x_unfold = F.unfold(x_offset, (D, D), stride=sf)

        # N, C_out * H_crop * W_crop, D ** 2
        x_final = x_unfold.view(N, C_out, D**2,
                                -1).permute(0, 1, 3, 2).reshape(N, -1, D**2)

        left_offset = 2 if sf in [3, 4] else 3
        # N, C_out, 1, H_crop, W_crop
        y_unfold = y[..., left_offset:H_crop + left_offset,
                     left_offset:W_crop + left_offset]

        # N, C_out * H_crop * W_crop, 1
        y_final = y_unfold.reshape(N, C_out * H_crop * W_crop, 1)

        # reg
        xtx = torch.bmm(x_final.permute(0, 2, 1), x_final)
        xtx += torch.diag(torch.ones(xtx.shape[-1])).unsqueeze(0).to(
            alpha.device) * alpha.squeeze(-1)

        xty = torch.bmm(x_final.permute(0, 2, 1), y_final)
        xty += d.reshape(N, -1, 1) * alpha.squeeze(-1)
        try:
            if stage == 0:
                xtx = xtx.view(1, -1, xtx.shape[2])
                xty = xty.view(1, -1, xty.shape[2])
                d = self.lstsqu_solve(xtx.cpu(), xty.cpu(),
                                      torch.tensor(N)).cuda()
                d = d.view(1, 1, 1, D, D).repeat(N, 1, 1, 1, 1)
            else:
                d = self.lstsq_solve(xtx.cpu(), xty.cpu()).cuda()
                d = d.view(N, 1, 1, D, D)
        except:
            pass

        return d

    def forward_conv(self,
                     x: torch.Tensor,
                     d: torch.Tensor,
                     y: torch.Tensor,
                     alpha: torch.Tensor,
                     sf: int = 1,
                     solve_kernel: bool = True):
        """
            x (dictionary): N, 1, C_in, H, W
            x (kernel): N, C_out, 1, H, W
            d: N, C_out, C_in, D, D
            y: N, C_out, 1, H, W

            alpha: N, 1, 1, 1
            reg: float
        """
        N, _, C_in, H, W = x.shape
        alpha = alpha * H * W

        if not solve_kernel:
            C_out = y.shape[1]
        else:
            C_out = 1

        D = d.shape[-1]

        # solve xtx
        # conv
        # xtx_raw: N, sf ** 2 * C_in, C_in, 2 * D - 1, 2 * D - 1
        xtx_raw = self.cal_xtx(x, D, sf=sf)

        # unfold
        # xtx_unfold_raw: N, sf ** 2 * C_in, C_in * D ** 2, D ** 2
        xtx_unfold_raw = unfold5d(xtx_raw, D)

        if sf > 1:
            # xtx_unfold_raw: N, sf ** 2, C_in, C_in * D ** 2, D, D
            xtx_unfold_raw = xtx_unfold_raw.view(N, sf**2, C_in, C_in * D**2,
                                                 D, D)
            xtx_unfold_shape = list(xtx_unfold_raw.shape)

            xtx_unfold_shape[1] = 1

            # xtx_unfold_new: N, 1, C_in, C_in * D ** 2, D, D
            xtx_unfold = xtx_unfold_raw.new_zeros(xtx_unfold_shape)
            xtx_unfold_raw = xtx_unfold_raw.view(N, sf, sf, C_in, C_in * D**2,
                                                 D, D)

            offset = ((D - 1) // 2) % sf

            for i in range(sf):
                for j in range(sf):
                    i_index = (-i + offset + sf) % sf
                    j_index = (-j + offset + sf) % sf
                    xtx_unfold[:, 0, :, :, i::sf,
                               j::sf] = xtx_unfold_raw[:, i_index,
                                                       j_index, :, :, i::sf,
                                                       j::sf]
            # xtx_unfold_new: N, C_in, C_in * D ** 2, D ** 2
            xtx_unfold = xtx_unfold.view(N, C_in, C_in * D**2, D**2)
        else:
            # N, sf ** 2 * C_in, C_in * D ** 2, D ** 2
            xtx_unfold = xtx_unfold_raw

            # xtx: N, C_in, C_in, D ** 2, D ** 2
        xtx = xtx_unfold.view(N, C_in, C_in, D**2, D**2)

        # flip
        # xtx: not changed
        xtx = xtx.flip(dims=(-1, ))

        # permute
        # xtx: N, C_in, d ** 2, C_in,  D ** 2
        xtx = xtx.permute(0, 1, 3, 2, 4)

        # reshape
        # xtx: N, C_in * d ** 2, C_in * D ** 2
        xtx = xtx.reshape(-1, C_in * D**2, C_in * D**2)

        # add xtx
        xtx[:, range(C_in * D**2),
            range(C_in *
                  D**2)] = xtx[:, range(C_in * D**2),
                               range(C_in *
                                     D**2)] + alpha.squeeze(-1).squeeze(-1)
        # solve xty
        # xty: N, C_out, C_in, D, D
        xty = self.cal_xty(x, y, D, sf=sf)

        # reshape
        # xty: N, C_out, C_in * D ** 2
        xty = xty.reshape(N, C_out, C_in * D**2)

        # permute
        # xty: N, C_in * D ** 2, C_out
        xty = xty.permute(0, 2, 1)

        # reg xty
        xty += alpha.squeeze(-1) * d.reshape(N, C_out, C_in * D**2).permute(
            0, 2, 1)

        # solve
        try:
            d = self.lstsq_solve(xtx, xty)
            d = d.view(N, C_in, D, D, C_out).permute(0, 4, 1, 2, 3)

        except RuntimeError:
            try:
                d = self.lu_solve(xtx, xty)
                d = d.view(N, C_in, D, D, C_out).permute(0, 4, 1, 2, 3)
            except RuntimeError:
                pass
        # TODO END

        return d

    def cal_xtx(self,
                x: torch.Tensor,
                d_size: int,
                sf: int = 1) -> torch.Tensor:
        """
            x (dn): N, 1, C_in, H, W
            x (sr): N, C_out, 1, H, W
            d_size: kernel (d) size
        """
        N, C_out, C_in, H, W = x.shape

        padding = d_size - 1

        if x.shape[1] == 1:  # solve d
            x_kernel = x.view(N, C_in, 1, 1, H, W)

            x_kernel_split = x_kernel

            # x: N, 1, C_in, H, W
            # x_kernel: N, 1, C_in, H, W -> N, sf ** 2 * C_in, 1, 1, H, W
            # xtx: N, sf ** 2 * C_in, C_in, 2 * d - 1, 2 * d - 1
            xtx = conv3d(x, x_kernel_split, padding, sample_wise=True)
        else:  # solve K
            x_kernel = x.view(N, 1, C_out, H, W)

            if sf > 1:
                zeros_shape = list(x_kernel.shape)
                zeros_shape[1] = zeros_shape[1] * sf**2
                x_kernel_split = x_kernel.new_zeros(zeros_shape)

                count = 0
                for i in range(sf):
                    for j in range(sf):
                        x_kernel_split[:, count, :, i::sf,
                                       j::sf] = x_kernel[:, :, :, i::sf,
                                                         j::sf].squeeze()
                        count += 1
            else:
                x_kernel_split = x_kernel

            # N, 1, 1, d_size, d_size
            xtx = conv2d(x.squeeze(2),
                         x_kernel_split,
                         padding=padding,
                         sample_wise=True)

            xtx = xtx.unsqueeze(2)
        return xtx

    def cal_xty(self, x: torch.Tensor, y: torch.Tensor, D: int, sf: int = 1):
        """
            x: N, 1, C_in, H, W; or
            N, C_out, 1, H, W
            d_size: kernel (d) size
            y: N, C_out, 1, H, W
        """
        N, C_out, _, H, W = y.shape
        padding = (D - 1) // 2
        if x.shape[1] == 1:  # solve d
            # N, C_out, C_in, D, D
            xty = conv3d(x, y.unsqueeze(3), padding, sample_wise=True)
        else:  # solve k
            # N, 1, C_out, H, W
            y_kernel = y.view(N, 1, C_out, H, W)
            y_kernel = upsample(y_kernel, sf, mode='zero')

            # N, 1, D, D
            xty = conv2d(x.squeeze(2),
                         y_kernel,
                         padding=padding,
                         sample_wise=True)
            xty = xty.unsqueeze(1)

        return xty


class TailNet(nn.Module):
    def __init__(self):
        super(TailNet, self).__init__()

    def forward(self, x, d):
        if d is not None:
            y = conv2d(F.pad(x, [
                (d.size(-1) - 1) // 2,
            ] * 4, mode='circular'),
                       d,
                       sample_wise=True)

        else:
            y = x

        return y


class HyPaNet(nn.Module):
    def __init__(
        self,
        in_nc=1,
        nc=256,
        out_nc=8,
    ):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, nc, 1, padding=0, bias=True), nn.Sigmoid(),
            nn.Conv2d(nc, out_nc, 1, padding=0, bias=True), nn.Softplus())

    def forward(self, x):
        x = (x - 0.098) / 0.0566
        x = self.mlp(x) + 1e-6
        return x


class UDKE(nn.Module):
    def __init__(self,
                 n_iter: int = 1,
                 in_nc: int = 1,
                 nc_x: List[int] = [64, 128, 256, 512],
                 nb: int = 1,
                 k_size: int = 25,
                 **kargs: Any):
        super(UDKE, self).__init__()

        self.head: nn.Module = HeadNet(k_size=k_size)

        self.body: nn.Module = BodyNet(in_nc=in_nc + 1,
                                       nc_x=nc_x,
                                       nc_d=nc_x,
                                       nc_k=nc_x,
                                       out_nc=in_nc,
                                       nb=nb,
                                       multi_stage=n_iter > 1)
        self.tail = TailNet()

        self.hypa_list: nn.ModuleList = nn.ModuleList()
        for _ in range(n_iter):
            self.hypa_list.append(HyPaNet(in_nc=1, out_nc=6))

        self.n_iter = n_iter

    def forward(self, y: torch.Tensor, sigma: torch.Tensor, sf: int = 1):
        onesided = sf == 1

        # prepare y_
        y_ = rfft(upsample(y, sf), onesided=onesided).unsqueeze(2)

        # head_net
        x, d, k = self.head(y, sf=sf)

        dxs = []
        ks = []

        for i in range(self.n_iter):
            hypas = self.hypa_list[i](sigma)
            alpha_x = hypas[:, 0].unsqueeze(-1)
            beta_x = hypas[:, 1].unsqueeze(-1)
            alpha_k = hypas[:, 4].unsqueeze(-1)
            beta_k = hypas[:, 5].unsqueeze(-1)
            k, x, d = self.body(k=k,
                                x=x,
                                d=d,
                                y=y,
                                y_=y_,
                                alpha_k=alpha_k,
                                beta_k=beta_k,
                                alpha_x=alpha_x,
                                beta_x=beta_x,
                                stage=i,
                                sf=sf)
            dx = self.tail(x, d)
            dxs.append(dx)
            ks.append(k)
        return dxs, ks, d

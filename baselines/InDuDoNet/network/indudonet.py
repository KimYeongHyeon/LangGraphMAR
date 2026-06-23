"""
MICCAI2021: ``InDuDoNet: An Interpretable Dual Domain Network for CT Metal Artifact Reduction''
paper link: https://arxiv.org/pdf/2109.05298.pdf

Adapted for CTMAR data (512x512 images, 1000 views x 900 detectors).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from odl.contrib import torch as odl_torch
from .priornet import UNet
from .build_gemotry import initialization, build_gemotry

filter = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) / 9  # for initialization
filter = filter.unsqueeze(dim=0).unsqueeze(dim=0)


def build_projection_operators(anatomy='body'):
    """Build forward and adjoint projection operators for the given anatomy.

    Includes scale normalization so FP output matches the original InDuDoNet
    paper's DeepLesion geometry scale (FP(ones) ≈ 21.6).
    Our geometry produces FP(ones) ≈ 564.2 due to larger pixel size.
    """
    import numpy as np
    para_ini = initialization(anatomy=anatomy)
    fp = build_gemotry(para_ini)

    # Compute scale factor: normalize FP output to match original paper
    ones = np.ones((para_ini.param['nx_h'], para_ini.param['ny_h']), dtype=np.float32)
    fp_ones_max = float(np.asarray(fp(ones)).max())
    # Original paper's FP(ones).max() ≈ 21.6 (416x416, reso=0.037)
    ORIGINAL_FP_SCALE = 21.6348
    scale = ORIGINAL_FP_SCALE / fp_ones_max

    # Wrap operators with scaling
    raw_fp = odl_torch.OperatorModule(fp)
    raw_pT = odl_torch.OperatorModule(fp.adjoint)

    class ScaledFP(nn.Module):
        def __init__(self, op, s):
            super().__init__()
            self.op = op
            self.s = s
        def forward(self, x):
            return self.op(x) * self.s

    class ScaledBP(nn.Module):
        def __init__(self, op, s):
            super().__init__()
            self.op = op
            self.s = s
        def forward(self, x):
            return self.op(x) * self.s

    op_modfp = ScaledFP(raw_fp, scale)
    op_modpT = ScaledBP(raw_pT, scale)

    return op_modfp, op_modpT


class InDuDoNet(nn.Module):
    def __init__(self, args):
        super(InDuDoNet, self).__init__()
        self.S = args.S                            # Stage number S includes the initialization process
        self.iter = self.S - 1                     # not include the initialization process
        self.num_u = args.num_channel + 1         # concat extra 1 term
        self.num_f = args.num_channel + 2         # concat extra 2 terms
        self.T = args.T

        # Build projection operators for the specified anatomy
        anatomy = getattr(args, 'anatomy', 'body')
        self.op_modfp, self.op_modpT = build_projection_operators(anatomy)

        # stepsize
        self.eta1const = args.eta1
        self.eta2const = args.eta2
        self.eta1 = torch.Tensor([self.eta1const])
        self.eta2 = torch.Tensor([self.eta2const])
        self.eta1S = self.make_coeff(self.S, self.eta1)
        self.eta2S = self.make_coeff(self.S, self.eta2)

        # weight factor
        self.alphaconst = args.alpha
        self.alpha = torch.Tensor([self.alphaconst])
        self.alphaS = self.make_coeff(self.S, self.alpha)

        # priornet
        self.priornet = UNet(n_channels=2, n_classes=1, n_filter=32)

        # proxNet for initialization
        self.proxNet_X0 = CTnet(args.num_channel + 1, self.T)
        self.proxNet_S0 = Projnet(args.num_channel + 1, self.T)

        # proxNet for iterative process
        self.proxNet_Xall = self.make_Xnet(self.S, args.num_channel + 1, self.T)
        self.proxNet_Sall = self.make_Snet(self.S, args.num_channel + 1, self.T)

        # Initialization S-domain by convoluting on XLI and SLI, respectively
        self.CX_const = filter.expand(args.num_channel, 1, -1, -1)
        self.CX = nn.Parameter(self.CX_const, requires_grad=True)
        self.CS_const = filter.expand(args.num_channel, 1, -1, -1)
        self.CS = nn.Parameter(self.CS_const, requires_grad=True)

        self.bn = nn.BatchNorm2d(1)

    def make_coeff(self, iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1)
        coeff = nn.Parameter(data=const_f, requires_grad=True)
        return coeff

    def make_Xnet(self, iters, channel, T):
        layers = []
        for i in range(iters):
            layers.append(CTnet(channel, T))
        return nn.Sequential(*layers)

    def make_Snet(self, iters, channel, T):
        layers = []
        for i in range(iters):
            layers.append(Projnet(channel, T))
        return nn.Sequential(*layers)

    def forward(self, Xma, XLI, M, Sma, SLI, Tr):
        op_modfp = self.op_modfp
        op_modpT = self.op_modpT

        # save mid-updating results
        ListS = []
        ListX = []
        ListYS = []

        # Initialization with channel concatenation and detachment
        XZ00 = F.conv2d(XLI, self.CX, stride=1, padding=1)
        input_Xini = torch.cat((XLI, XZ00), dim=1)
        XZ_ini = self.proxNet_X0(input_Xini)
        X0 = XZ_ini[:, :1, :, :]
        XZ = XZ_ini[:, 1:, :, :]
        X = X0

        SZ00 = F.conv2d(SLI, self.CS, stride=1, padding=1)
        input_Sini = torch.cat((SLI, SZ00), dim=1)
        SZ_ini = self.proxNet_S0(input_Sini)
        S0 = SZ_ini[:, :1, :, :]
        SZ = SZ_ini[:, 1:, :, :]
        S = S0
        ListS.append(S)

        # PriorNet
        prior_input = torch.cat((Xma, XLI), dim=1)
        Xs = XLI + self.priornet(prior_input)
        Y = op_modfp(F.relu(self.bn(Xs)) / 255)
        Y = Y / 4.0 * 255

        # 1st iteration: Updating X0, S0-->S1
        PX = op_modfp(X / 255) / 4.0 * 255
        GS = Y * (Y * S - PX) + self.alphaS[0] * Tr * Tr * Y * (Y * S - Sma)
        S_next = S - self.eta1S[0] / 10 * GS
        inputS = torch.cat((S_next, SZ), dim=1)
        outS = self.proxNet_Sall[0](inputS)
        S = outS[:, :1, :, :]
        SZ = outS[:, 1:, :, :]
        ListS.append(S)
        ListYS.append(Y * S)

        # 1st iteration: Updating X0, S1-->X1
        ESX = PX - Y * S
        GX = op_modpT((ESX / 255) * 4.0)
        X_next = X - self.eta2S[0] / 10 * GX
        inputX = torch.cat((X_next, XZ), dim=1)
        outX = self.proxNet_Xall[0](inputX)
        X = outX[:, :1, :, :]
        XZ = outX[:, 1:, :, :]
        ListX.append(X)

        for i in range(self.iter):
            # updating S
            PX = op_modfp(X / 255) / 4.0 * 255
            GS = Y * (Y * S - PX) + self.alphaS[i + 1] * Tr * Tr * Y * (Y * S - Sma)
            S_next = S - self.eta1S[i + 1] / 10 * GS
            inputS = torch.cat((S_next, SZ), dim=1)
            outS = self.proxNet_Sall[i + 1](inputS)
            S = outS[:, :1, :, :]
            SZ = outS[:, 1:, :, :]
            ListS.append(S)
            ListYS.append(Y * S)

            # updating X
            ESX = PX - Y * S
            GX = op_modpT((ESX / 255) * 4.0)
            X_next = X - self.eta2S[i + 1] / 10 * GX
            inputX = torch.cat((X_next, XZ), dim=1)
            outX = self.proxNet_Xall[i + 1](inputX)
            X = outX[:, :1, :, :]
            XZ = outX[:, 1:, :, :]
            ListX.append(X)
        return ListX, ListS, ListYS


# proxNet_S
class Projnet(nn.Module):
    def __init__(self, channel, T):
        super(Projnet, self).__init__()
        self.channels = channel
        self.T = T
        self.layer = self.make_resblock(self.T)

    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(
                nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              nn.BatchNorm2d(self.channels),
                              nn.ReLU(),
                              nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              nn.BatchNorm2d(self.channels),
                              ))
        return nn.Sequential(*layers)

    def forward(self, input):
        S = input
        for i in range(self.T):
            S = F.relu(S + self.layer[i](S))
        return S


# proxNet_X
class CTnet(nn.Module):
    def __init__(self, channel, T):
        super(CTnet, self).__init__()
        self.channels = channel
        self.T = T
        self.layer = self.make_resblock(self.T)

    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(),
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(self.channels),
            ))
        return nn.Sequential(*layers)

    def forward(self, input):
        X = input
        for i in range(self.T):
            X = F.relu(X + self.layer[i](X))
        return X

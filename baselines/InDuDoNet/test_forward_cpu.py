"""
Test InDuDoNet forward pass with FIXED h5 data on CPU.
Patches astra_cuda -> astra_cpu for H100 compatibility.
"""
import sys
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Monkey-patch odl.tomo.RayTransform BEFORE anything imports the network modules
import odl
import odl.tomo
import odl.tomo.operators.ray_trafo

_OrigRayTransformInit = odl.tomo.operators.ray_trafo.RayTransform.__init__

def _patched_init(self, *args, **kwargs):
    if kwargs.get('impl') == 'astra_cuda':
        print(f"[PATCH] Redirecting astra_cuda -> astra_cpu")
        kwargs['impl'] = 'astra_cpu'
    return _OrigRayTransformInit(self, *args, **kwargs)

odl.tomo.operators.ray_trafo.RayTransform.__init__ = _patched_init

print("[PATCH] odl.tomo.RayTransform.__init__ patched to use astra_cpu")

# Now safe to import everything
import numpy as np
import torch
import torch.nn.functional as F
from odl.contrib import torch as odl_torch
from deeplesion.Dataset import MARTrainDataset
from network.indudonet import InDuDoNet

DATA_PATH = '../../data/train/indudonet/head_v2/'

print("\n" + "="*70)
print("STEP 1: Load trainmask and create dataset")
print("="*70)

train_mask = np.load(os.path.join(DATA_PATH, 'trainmask.npy'))
print(f"trainmask shape: {train_mask.shape}, dtype: {train_mask.dtype}")
print(f"trainmask range: [{train_mask.min():.4f}, {train_mask.max():.4f}]")

dataset = MARTrainDataset(DATA_PATH, 416, train_mask)
print(f"Dataset size: {len(dataset)}")

print("\n" + "="*70)
print("STEP 2: Load one sample and inspect tensors")
print("="*70)

sample = dataset[0]
names = ['Xma', 'XLI', 'Xgt', 'Mask', 'Sma', 'SLI', 'Sgt', 'Tr']
for name, tensor in zip(names, sample):
    print(f"  {name:6s}: shape={str(tensor.shape):20s} dtype={tensor.dtype}  "
          f"range=[{tensor.min().item():.4f}, {tensor.max().item():.4f}]")

Xma, XLI, Xgt, Mask, Sma, SLI, Sgt, Tr = sample

print(f"\n  NOTE: metal_trace from Dataset is in IMAGE domain {Tr.shape}")
print(f"  But InDuDoNet forward pass needs Tr in SINOGRAM domain (1, 640, 641)")
print(f"  The FIXED h5 data has metal_trace at (416,416) -- this is WRONG for InDuDoNet.")
print(f"  Original DeepLesion data had metal_trace at (640,641) in sinogram domain.")
print(f"\n  WORKAROUND: Forward-project binary metal mask to get sinogram-domain trace")

# Forward project the metal mask to sinogram domain
import network.indudonet as ind_mod
# Tr from Dataset is already (1 - original_trace), so undo that
metal_mask_img = 1.0 - Tr[0].numpy()  # shape (416, 416), 1 where metal, 0 elsewhere
# Forward project
from deeplesion.Dataset import ray_trafo as ds_ray_trafo
metal_sino = np.asarray(ds_ray_trafo(metal_mask_img))  # shape (640, 641)
# Threshold to binary: any ray that passes through metal
metal_trace_sino = (metal_sino > 0.01).astype(np.float32)
# Apply same transform as Dataset: Tr = 1 - trace, then expand dims
Tr_sino = 1.0 - metal_trace_sino
Tr_sino = np.transpose(np.expand_dims(Tr_sino, 2), (2, 0, 1))
Tr = torch.Tensor(Tr_sino)
print(f"  Tr (sinogram domain): shape={Tr.shape}, range=[{Tr.min():.4f}, {Tr.max():.4f}]")

# Add batch dimension
Xma = Xma.unsqueeze(0)
XLI = XLI.unsqueeze(0)
Xgt = Xgt.unsqueeze(0)
Mask = Mask.unsqueeze(0)
Sma = Sma.unsqueeze(0)
SLI = SLI.unsqueeze(0)
Sgt = Sgt.unsqueeze(0)
Tr = Tr.unsqueeze(0)

print("\nBatched shapes:")
for name, tensor in zip(names, [Xma, XLI, Xgt, Mask, Sma, SLI, Sgt, Tr]):
    print(f"  {name:6s}: {tensor.shape}")

print("\n" + "="*70)
print("STEP 3: Create InDuDoNet model on CPU")
print("="*70)

args = argparse.Namespace(
    S=10,
    num_channel=32,
    T=4,
    eta1=1,
    eta2=5,
    alpha=0.5,
)

model = InDuDoNet(args)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model created on CPU, total parameters: {num_params:,}")

print("\n" + "="*70)
print("STEP 4: Forward pass")
print("="*70)

try:
    model.train()
    # forward(Xma, XLI, M, Sma, SLI, Tr)
    ListX, ListS, ListYS = model(Xma, XLI, Mask, Sma, SLI, Tr)
    print("FORWARD PASS SUCCEEDED!")
    print(f"  ListX length: {len(ListX)}")
    print(f"  ListS length: {len(ListS)}")
    print(f"  ListYS length: {len(ListYS)}")
    for i, x in enumerate(ListX):
        print(f"  ListX[{i}]: shape={x.shape}, range=[{x.min().item():.4f}, {x.max().item():.4f}]")
    for i, s in enumerate(ListS):
        print(f"  ListS[{i}]: shape={s.shape}, range=[{s.min().item():.4f}, {s.max().item():.4f}]")
    for i, ys in enumerate(ListYS):
        print(f"  ListYS[{i}]: shape={ys.shape}, range=[{ys.min().item():.4f}, {ys.max().item():.4f}]")
except Exception as e:
    print(f"FORWARD PASS FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("STEP 5: Loss computation and backward pass")
print("="*70)

try:
    gamma = 0.1
    S = args.S

    # Loss as in train.py
    loss_l2YSmid = 0.1 * F.mse_loss(ListYS[S - 2], Sgt)
    loss_l2Xmid = 0.1 * F.mse_loss(ListX[S - 2] * (1 - Mask), Xgt * (1 - Mask))
    loss_l2YSf = F.mse_loss(ListYS[-1], Sgt)
    loss_l2Xf = F.mse_loss(ListX[-1] * (1 - Mask), Xgt * (1 - Mask))
    loss_l2YS = loss_l2YSf + loss_l2YSmid
    loss_l2X = loss_l2Xf + loss_l2Xmid
    loss = gamma * loss_l2YS + loss_l2X
    print(f"  loss_l2YS: {loss_l2YS.item():.6f}")
    print(f"  loss_l2X:  {loss_l2X.item():.6f}")
    print(f"  total loss: {loss.item():.6f}")

    loss.backward()
    print("BACKWARD PASS SUCCEEDED!")

    # Check gradients exist
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"  Parameters with gradients: {grad_count}/{total_params}")

except Exception as e:
    print(f"LOSS/BACKWARD FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED")
print("="*70)

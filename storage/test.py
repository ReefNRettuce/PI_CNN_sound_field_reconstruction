import torch
from helmholtz import Helmholtz_Loss

wavenumber_k = 5.5
grid_spacing_l = 0.1

helmholtz_fn = Helmholtz_Loss(wavenumber_k, grid_spacing_l)

# Test 1: zeros
x = torch.zeros(1, 8, 32, 32)
print(f"test 1 All zeros: {helmholtz_fn(x).item()}")

# Test 2: small pressure, zero derivatives
x = torch.zeros(1, 8, 32, 32)
x[:, 0] = 0.1  # small constant pressure (real)
x[:, 4] = 0.1  # small constant pressure (imag)
print(f"test 2 Constant pressure, zero derivatives: {helmholtz_fn(x).item()}")

# Test 2b: smooth varying pressure, zero derivatives
x = torch.zeros(1, 8, 32, 32)
coords = torch.linspace(0, 1, 32)
smooth_field = torch.sin(coords).unsqueeze(0).unsqueeze(0) * 0.1
x[:, 0] = smooth_field
x[:, 4] = smooth_field
print(f"Smooth pressure, zero derivatives: {helmholtz_fn(x).item()}")

# Test 3: small random pressure, zero derivatives
x = torch.zeros(1, 8, 32, 32)
x[:, 0] = torch.randn(1, 32, 32) * 0.1
x[:, 4] = torch.randn(1, 32, 32) * 0.1
print(f"test 3 Random pressure, zero derivatives: {helmholtz_fn(x).item()}")

# Test 4: all channels small random
x = torch.randn(1, 8, 32, 32) * 0.1
print(f"test 4 All channels random (0.1 scale): {helmholtz_fn(x).item()}")

# Test 5: realistic scale from your network
x = torch.randn(1, 8, 32, 32) * 0.14
print(f"test 5 All channels random (0.14 scale): {helmholtz_fn(x).item()}")

# Test 6: testing Helmholtz function

import torch
from torch.utils.data import DataLoader
from helmholtz import Helmholtz_Loss
from train_v3a import RoomAcousticDataset, WAVENUMBER_K, GRID_SPACING

# Load one batch
dataset = RoomAcousticDataset(mode='train')
loader = DataLoader(dataset, batch_size=1, shuffle=False)
sample = next(iter(loader))

target = sample['target']  # [1, 2, 32, 32] - ground truth pressure

# Build 8-channel input for Helmholtz (pressure + zero derivatives)
helmholtz_input = torch.zeros(1, 8, 32, 32)
helmholtz_input[:, 0] = target[:, 0]  # real pressure
helmholtz_input[:, 4] = target[:, 1]  # imag pressure
# Channels 1-3, 5-7 stay zero

helmholtz_fn = Helmholtz_Loss(WAVENUMBER_K, GRID_SPACING)
loss = helmholtz_fn(helmholtz_input)

print(f"Ground truth Helmholtz residual: {loss.item()}")

import torch
import numpy as np
from helmholtz import Helmholtz_Loss

# Parameters
k = 5.5  # wavenumber for 300 Hz
l = 3.0 / 31  # grid spacing

helmholtz_fn = Helmholtz_Loss(k, l)

# Test 1: All zeros (should be exactly 0)
x = torch.zeros(1, 8, 32, 32)
print(f"Test 1 - Zeros: {helmholtz_fn(x).item()}")

# Test 2: Constant pressure (should be small, order 10-100)
x = torch.zeros(1, 8, 32, 32)
x[:, 0, :, :] = 1.0  # constant real pressure
x[:, 4, :, :] = 0.0  # zero imag pressure
print(f"Test 2 - Constant pressure: {helmholtz_fn(x).item()}")

# Test 3: Plane wave that actually satisfies Helmholtz
# u = cos(kx * x) where kx = k (so ∇²u + k²u = 0)
x = torch.zeros(1, 8, 32, 32)
grid = torch.linspace(0, 3.0, 32)
kx = k  # wavenumber in x direction

for i in range(32):
    for j in range(32):
        pos_x = grid[j]
        pos_y = grid[i]
        
        # u = cos(kx * x)
        x[0, 0, i, j] = np.cos(kx * pos_x)       # u
        x[0, 1, i, j] = -kx * np.sin(kx * pos_x) # du/dx
        x[0, 2, i, j] = 0                         # du/dy
        x[0, 3, i, j] = 0                         # d²u/dxdy

print(f"Test 3 - Plane wave (should be ~0): {helmholtz_fn(x).item()}")

# Test 4: Print intermediate values
x = torch.zeros(1, 8, 32, 32)
x[:, 0, :, :] = 1.0

# Manually check what's happening
print(f"\nDebug info:")
print(f"k = {k}, k² = {k**2}, k⁴ = {k**4}")
print(f"l = {l}")
print(f"area = {((32-1)*l)**2}")
print(f"sum(C1) = {torch.sum(helmholtz_fn.c1).item()}")
print(f"sum(C2) = {torch.sum(helmholtz_fn.c2).item()}")
print(f"sum(C3) = {torch.sum(helmholtz_fn.c3).item()}")


print("Last test disregard everything else")


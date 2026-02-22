import scipy.io
import torch
from helmholtz import Helmholtz_Loss
import numpy as np

# Load one MATLAB file
mat_data = scipy.io.loadmat(('/Users/leifefrancisco/Documents/Workspace/PI_CNN_V2/PI_CNN_V3a/V3_a_a_0_001/data/room_acoustic_data_room_0001.mat'))
u_grid = mat_data['u_grid']

print(f"u_grid shape: {u_grid.shape}")
print(f"u_grid dtype: {u_grid.dtype}")
print(f"u_grid max: {np.abs(u_grid).max()}")

# Check what k MATLAB used
print(f"MATLAB k: {mat_data['k']}")

# Build Helmholtz input with just pressure, zero derivatives
helmholtz_input = torch.zeros(1, 8, 32, 32)
helmholtz_input[0, 0] = torch.from_numpy(u_grid.real).float()
helmholtz_input[0, 4] = torch.from_numpy(u_grid.imag).float()

k = 5.5
l = 3.0 / 31
helmholtz_fn = Helmholtz_Loss(k, l)
loss = helmholtz_fn(helmholtz_input)

print(f"MATLAB data Helmholtz loss: {loss.item()}")

# Compute derivatives using finite differences
du_dx = np.gradient(u_grid.real, l, axis=1)  # derivative in x
du_dy = np.gradient(u_grid.real, l, axis=0)  # derivative in y
du_dxdy = np.gradient(du_dx, l, axis=0)      # cross derivative

# Same for imaginary part
du_dx_imag = np.gradient(u_grid.imag, l, axis=1)
du_dy_imag = np.gradient(u_grid.imag, l, axis=0)
du_dxdy_imag = np.gradient(du_dx_imag, l, axis=0)

# Build full 8-channel input
helmholtz_input = torch.zeros(1, 8, 32, 32)
helmholtz_input[0, 0] = torch.from_numpy(u_grid.real).float()
helmholtz_input[0, 1] = torch.from_numpy(du_dx).float()
helmholtz_input[0, 2] = torch.from_numpy(du_dy).float()
helmholtz_input[0, 3] = torch.from_numpy(du_dxdy).float()
helmholtz_input[0, 4] = torch.from_numpy(u_grid.imag).float()
helmholtz_input[0, 5] = torch.from_numpy(du_dx_imag).float()
helmholtz_input[0, 6] = torch.from_numpy(du_dy_imag).float()
helmholtz_input[0, 7] = torch.from_numpy(du_dxdy_imag).float()

loss = helmholtz_fn(helmholtz_input)
print(f"With derivatives: {loss.item()}")

import numpy as np

# Your grid spacing
l = 3.0 / 32

# Check derivative magnitudes
du_dx = np.gradient(u_grid.real, l, axis=1)
du_dy = np.gradient(u_grid.real, l, axis=0)

print(f"Pressure max: {np.abs(u_grid).max()}")
print(f"du/dx max: {np.abs(du_dx).max()}")
print(f"du/dy max: {np.abs(du_dy).max()}")
print(f"Expected derivative magnitude (k * u_max): {5.5 * 1.0}")


u_grid = u_grid.T  # Transpose to match Python convention
# Then compute derivatives...

# Maybe axes are swapped?
du_dx = np.gradient(u_grid.real, l, axis=0)  # swapped
du_dy = np.gradient(u_grid.real, l, axis=1)  # swapped
du_dxdy = np.gradient(du_dx, l, axis=1)      # swapped

du_dx_imag = np.gradient(u_grid.imag, l, axis=0)
du_dy_imag = np.gradient(u_grid.imag, l, axis=1)
du_dxdy_imag = np.gradient(du_dx_imag, l, axis=1)

# Rebuild and test
helmholtz_input = torch.zeros(1, 8, 32, 32)
helmholtz_input[0, 0] = torch.from_numpy(u_grid.real).float()
helmholtz_input[0, 1] = torch.from_numpy(du_dx).float()
helmholtz_input[0, 2] = torch.from_numpy(du_dy).float()
helmholtz_input[0, 3] = torch.from_numpy(du_dxdy).float()
helmholtz_input[0, 4] = torch.from_numpy(u_grid.imag).float()
helmholtz_input[0, 5] = torch.from_numpy(du_dx_imag).float()
helmholtz_input[0, 6] = torch.from_numpy(du_dy_imag).float()
helmholtz_input[0, 7] = torch.from_numpy(du_dxdy_imag).float()

loss = helmholtz_fn(helmholtz_input)
print(f"With swapped axes: {loss.item()}")
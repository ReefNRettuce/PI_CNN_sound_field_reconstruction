# train_v3a.py #This is now 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from model_v3a import tiny_unet_small
from helmholtz import Helmholtz_Loss

# ========== CONFIGURATION ==========
EXPERIMENT_NAME = 'V3c'
PARENT_DIRECTORY = ''

# Standardized Paths
DATA_PATH = ''
RESULTS_PATH = f" "

# File Outputs
MODEL_SAVE_FILE = f'{RESULTS_PATH}/best_model.pth'
LOSS_PLOT_FILE = f"{RESULTS_PATH}/loss_curve.png"
TRAINING_LOG_FILE = f"{RESULTS_PATH}/training_log.txt"

# Hyperparameters
ROOM_LENGTH = 3.0 
GRID_SIZE = 32
GRID_SPACING = ROOM_LENGTH / (GRID_SIZE - 1)
FREQUENCY = 300
SOUND_SPEED = 343
WAVENUMBER_K = 2 * np.pi * FREQUENCY / SOUND_SPEED 

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
MAX_EPOCHS = 5000
PATIENCE = 1000

# Resume from data-only checkpoint (set True if physics caused collapse)
RESUME_FROM_DATA_ONLY = False

# ========== DEVICE SETUP ==========
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ MPS device found.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ CUDA device found.")
else:
    device = torch.device("cpu")
    print("! Using CPU.")

# Ensure Results Directory Exists
Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)

# ========== DATASET CLASS ==========
class RoomAcousticDataset(Dataset):
    def __init__(self, data_dir=DATA_PATH, mode='train'):
        self.data_dir = Path(data_dir)
        self.file_list = sorted(list(self.data_dir.glob('room_acoustic_data_*.mat')))
        
        if len(self.file_list) == 0:
            print(f"ERROR: No files found in {data_dir}.")
            exit()

        split_idx = int(0.8 * len(self.file_list))
        if mode == 'train':
            self.file_list = self.file_list[:split_idx]
        else:
            self.file_list = self.file_list[split_idx:]
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        mat_data = scipy.io.loadmat(self.file_list[idx])
        
        # Extract ground truth from MATLAB data
        ground_truth = mat_data['u_grid']  # Shape: (32, 32), complex
        
        # Create mask (fixed sensor positions)
        mask = np.zeros((32, 32))
        sensor_indices = []
        sensors_per_side = 4  # 4x4 = 16 sensors
        for i in range(sensors_per_side):
            for j in range(sensors_per_side):
                x_idx = int((i + 1) * 32 / (sensors_per_side + 1))
                y_idx = int((j + 1) * 32 / (sensors_per_side + 1))
                sensor_indices.append((x_idx, y_idx))
                mask[x_idx, y_idx] = 1.0
        
        # Extract measurements at sensor locations
        measurements = np.array([ground_truth[si, sj] for (si, sj) in sensor_indices])
        
        # Normalize to [-1, 1]
        if np.iscomplexobj(ground_truth):
            max_val = max(np.max(np.abs(ground_truth.real)), np.max(np.abs(ground_truth.imag)))
        else:
            max_val = np.max(np.abs(ground_truth))
        
        scale_factor = max_val if max_val > 0 else 1.0
        
        # Build input tensor
        real_input = np.zeros((32, 32), dtype=np.float32)
        imag_input = np.zeros((32, 32), dtype=np.float32)
        
        if np.iscomplexobj(measurements):
            real_input[mask == 1] = measurements.real / scale_factor
            imag_input[mask == 1] = measurements.imag / scale_factor
        else:
            real_input[mask == 1] = measurements / scale_factor

        input_tensor = torch.from_numpy(np.stack([real_input, imag_input])).float()
        
        # Build target tensor
        if np.iscomplexobj(ground_truth):
            real_gt = ground_truth.real / scale_factor
            imag_gt = ground_truth.imag / scale_factor
        else:
            real_gt = ground_truth
            imag_gt = np.zeros((32, 32), dtype=np.float32)
            
        target_tensor = torch.from_numpy(np.stack([real_gt, imag_gt])).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        
        return {'input': input_tensor, 'mask': mask_tensor, 'target': target_tensor}

# ========== TRAINING ==========
def train_one_epoch(dataloader, model, mse_loss_fn, helmholtz_fn, optimizer, device, physics_weight):
    model.train()
    total_loss = 0
    physics_loss_list = []
    data_loss_list = []

    for batch in dataloader:
        inputs = batch['input'].to(device)
        masks = batch['mask'].to(device)
        targets = batch['target'].to(device)
        
        optimizer.zero_grad()
        full_outputs = model(inputs, masks)
        
        # MSE Loss (only on pressure channels 0 and 4)
        pred_pressure = torch.cat([full_outputs[:, 0:1], full_outputs[:, 4:5]], dim=1)
        data_loss = mse_loss_fn(pred_pressure, targets)
        data_loss_list.append(data_loss.item())
        
        # Physics Loss
        physics_loss = helmholtz_fn(full_outputs)
        physics_loss_list.append(physics_loss.item())
        
        # Total Loss
        loss = data_loss + (physics_weight * physics_loss)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return (total_loss / len(dataloader)), np.mean(physics_loss_list), np.mean(data_loss_list)

def validate(dataloader, model, mse_loss_fn, helmholtz_fn, device, physics_weight):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            masks = batch['mask'].to(device)
            targets = batch['target'].to(device)
            
            full_outputs = model(inputs, masks)
            pred_pressure = torch.cat([full_outputs[:, 0:1], full_outputs[:, 4:5]], dim=1)
            
            data_loss = mse_loss_fn(pred_pressure, targets)
            physics_loss = helmholtz_fn(full_outputs)
            
            total_loss += (data_loss + (physics_weight * physics_loss)).item()
            
    return total_loss / len(dataloader)

def plot_losses(train_losses, val_losses, physics_losses, data_losses):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total loss
    axes[0, 0].plot(train_losses, label='Train')
    axes[0, 0].plot(val_losses, label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Data loss
    axes[0, 1].plot(data_losses, label='Data Loss', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].set_title('Data Loss (MSE)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Physics loss
    axes[1, 0].plot(physics_losses, label='Physics Loss', color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Helmholtz Loss')
    axes[1, 0].set_title('Physics Loss (Helmholtz)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Log scale total loss
    axes[1, 1].semilogy(train_losses, label='Train')
    axes[1, 1].semilogy(val_losses, label='Val')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Total Loss (log scale)')
    axes[1, 1].set_title('Total Loss (Log Scale)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_FILE)
    print(f'✓ Saved loss curves to {LOSS_PLOT_FILE}')
    plt.close()

# ========== MAIN ==========
def main():
    print(f'Experiment: {EXPERIMENT_NAME}')
    print(f'Results will be saved to: {RESULTS_PATH}')
    
    train_dataset = RoomAcousticDataset(mode='train')
    val_dataset = RoomAcousticDataset(mode='val')
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = tiny_unet_small(in_channels=2, out_channels=8).to(device)
    
    # Resume from checkpoint if requested
    if RESUME_FROM_DATA_ONLY:
        checkpoint_path = f"{RESULTS_PATH}/checkpoint_data_only.pth"
        if Path(checkpoint_path).exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"✓ Loaded data-only checkpoint from {checkpoint_path}")
        else:
            print(f"WARNING: Checkpoint not found at {checkpoint_path}, starting fresh")
    
    mse_loss = nn.MSELoss()
    helmholtz_fn = Helmholtz_Loss(wavenumber_k=WAVENUMBER_K, grid_spacing_l=GRID_SPACING).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    
    train_losses = []
    val_losses = []
    physics_losses = []
    data_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Clear training log
    with open(TRAINING_LOG_FILE, "w") as f:
        f.write(f"Training started for {EXPERIMENT_NAME}\n")
        f.write(f"{'='*60}\n")
    
    print('\nStarting Training...')
    print(f'Max epochs: {MAX_EPOCHS}, Patience: {PATIENCE}')
    print(f'{'='*60}')
    
    for epoch in range(MAX_EPOCHS):
        
        # ========== PHYSICS WEIGHT SCHEDULER ==========
        if epoch < 50:
            current_physics_weight = 0.0
        elif epoch == 50:
            # Save data-only checkpoint before introducing physics
            torch.save(model.state_dict(), f"{RESULTS_PATH}/checkpoint_data_only.pth")
            print(f"✓ Saved data-only checkpoint at epoch {epoch+1}")
            current_physics_weight = 1e-7
        elif epoch < 500:
            # Logarithmic ramp from 1e-9 to 1e-4 over epochs 50-500
            progress = (epoch - 50) / 450
            current_physics_weight = 10 ** (-7 + 4 * progress)
        else:
            current_physics_weight = 1e-3
        
        # ========== TRAIN AND VALIDATE ==========
        t_loss, phy_loss_mean, d_loss_mean = train_one_epoch(
            train_loader, model, mse_loss, helmholtz_fn, optimizer, device, current_physics_weight
        )
        v_loss = validate(val_loader, model, mse_loss, helmholtz_fn, device, current_physics_weight)
        
        scheduler.step()
        
        # ========== RECORD LOSSES ==========
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        physics_losses.append(phy_loss_mean)
        data_losses.append(d_loss_mean)
        
        # ========== LOGGING ==========
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}: Train={t_loss:.6f}, Val={v_loss:.6f}, '
              f'Data={d_loss_mean:.6f}, Phys={phy_loss_mean:.6f}, '
              f'Lambda={current_physics_weight:.2e}, LR={current_lr:.2e}')
        
        # Log to file
        with open(TRAINING_LOG_FILE, "a") as f:
            f.write(f"Epoch {epoch+1}: Train={t_loss:.6f}, Val={v_loss:.6f}, "
                    f"Data={d_loss_mean:.6f}, Phys={phy_loss_mean:.6f}, "
                    f"Lambda={current_physics_weight:.2e}, LR={current_lr:.2e}\n")
        
        # ========== CHECKPOINTING ==========
        # Save every 500 epochs
        if (epoch + 1) % 500 == 0:
            torch.save(model.state_dict(), f"{RESULTS_PATH}/checkpoint_epoch_{epoch+1}.pth")
            print(f"✓ Saved checkpoint at epoch {epoch+1}")
            # Also save loss plot periodically
            plot_losses(train_losses, val_losses, physics_losses, data_losses)
        
        # Save best model
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_FILE)
        else:
            patience_counter += 1
        
        # ========== EARLY STOPPING ==========
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # ========== FINAL SAVE ==========
    plot_losses(train_losses, val_losses, physics_losses, data_losses)
    torch.save(model.state_dict(), f"{RESULTS_PATH}/final_model.pth")
    print(f'\n✓ Training complete!')
    print(f'Best validation loss: {best_val_loss:.6f}')

if __name__ == '__main__':
    main()

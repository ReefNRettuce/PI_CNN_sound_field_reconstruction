# analyze_dataset_v3a.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd 
import scipy.io

from model_v3a import tiny_unet_small

# ========== CONFIGURATION ==========
VERSION_NUMBER = "V3a"
EXPERIMENT_NAME = "V3a_a_0_001" 
PARENT_DIRECTORY = f"PI_CNN_{VERSION_NUMBER}"

# Standardized Paths
BASE_PATH = f"./{PARENT_DIRECTORY}/{EXPERIMENT_NAME}"
DATA_PATH = '/Users/leifefrancisco/Documents/Workspace/PI_CNN_V2/PI_CNN_V3a/V3a_a_0_001/data'
RESULTS_PATH = f"{BASE_PATH}/results"

# Inputs
MODEL_PATH = '/Users/leifefrancisco/Documents/Workspace/PI_CNN_V2/PI_CNN_V3a/V3a_a_0_001/results/best_model.pth'

# Outputs
CSV_REPORT_FILE = f"{RESULTS_PATH}/reconstruction_report.csv"
HISTOGRAM_FILE = f"{RESULTS_PATH}/psnr_histograms.png"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if torch.cuda.is_available(): DEVICE = torch.device("cuda")

# Ensure Output Dir Exists
Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)

# ========== DATASET CLASS ==========
class RoomAcousticDataset(Dataset):
    def __init__(self, data_dir, mode='val'):
        self.data_dir = Path(data_dir)
        self.file_list = sorted(list(self.data_dir.glob('room_acoustic_data_room_*.mat')))
        
        if len(self.file_list) == 0:
            print(f"ERROR: No .mat files found in {data_dir}")
            exit()
        
        split_idx = int(0.8 * len(self.file_list))
        if mode == 'train':
            self.file_list = self.file_list[:split_idx]
        elif mode == "all":
            self.file_list = self.file_list[:]
        else:
            self.file_list = self.file_list[split_idx:]
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        mat_data = scipy.io.loadmat(file_path)
        
        # Load MATLAB variables
        ground_truth = mat_data['u_grid']  # Complex pressure field
        source_x = mat_data['source_x'].item()
        source_y = mat_data['source_y'].item()
        
        # Create mask (fixed sensor positions)
        mask = np.zeros((32, 32), dtype=np.float32)
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
            for k, (si, sj) in enumerate(sensor_indices):
                real_input[si, sj] = measurements[k].real / scale_factor
                imag_input[si, sj] = measurements[k].imag / scale_factor
        else:
            for k, (si, sj) in enumerate(sensor_indices):
                real_input[si, sj] = measurements[k] / scale_factor

        input_tensor = torch.from_numpy(np.stack([real_input, imag_input])).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        
        # Normalize ground truth for comparison
        gt_normalized = ground_truth / scale_factor

        return {
            'input': input_tensor,
            'mask': mask_tensor,
            'gt_complex': gt_normalized,
            'filename': file_path.name,
            'source_x': source_x,
            'source_y': source_y,
            'scale_factor': scale_factor
        }

# ========== METRICS ==========
def calculate_psnr(pred, target, max_val=1.0):
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10: return 100.0
    return 20 * np.log10(max_val / np.sqrt(mse))

# ========== MAIN ANALYSIS ==========
def run_full_analysis():
    print(f"Loading model from {MODEL_PATH}...")
    model = tiny_unet_small(in_channels=2, out_channels=8).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model not found at {MODEL_PATH}")
        return
        
    model.eval()
    val_dataset = RoomAcousticDataset(DATA_PATH, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # === ADD THIS BLOCK ===
    print("Checking for mode collapse...")
    outputs_list = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 10: break
            outputs = model(batch['input'].to(DEVICE), batch['mask'].to(DEVICE))
            outputs_list.append(outputs[0, 0].cpu().numpy())
    
    stacked = np.stack(outputs_list)
    print(f"Output variance across samples: {np.var(stacked):.6f}")
    print(f"Mean output range: {stacked.mean(axis=(1,2)).std():.6f}")
    
    if np.var(stacked) < 0.001:
        print("WARNING: Very low variance - possible mode collapse!")
    # === END BLOCK ===
    
    print(f"Analyzing {len(val_dataset)} samples...")
    results = []
    results = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch['input'].to(DEVICE)
            masks = batch['mask'].to(DEVICE)
            gt_complex = batch['gt_complex'].numpy()[0]
            
            outputs = model(inputs, masks)
            
            pred_real = outputs[0, 0, :, :].cpu().numpy()
            pred_imag = outputs[0, 4, :, :].cpu().numpy()
            pred_complex = pred_real + 1j * pred_imag
            
            psnr_mag = calculate_psnr(np.abs(pred_complex), np.abs(gt_complex))
            
            results.append({
                'filename': batch['filename'][0],
                'source_x': batch['source_x'].item(),
                'source_y': batch['source_y'].item(),
                'PSNR_Mag': psnr_mag
            })
            
            if (i+1) % 50 == 0: print(f"Processed {i+1}...")

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(CSV_REPORT_FILE, index=False)
    print(f"\n✓ Report saved to: {CSV_REPORT_FILE}")
    print(df.describe())

    # Plot Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(df['PSNR_Mag'], bins=20, color='lightgreen', edgecolor='black')
    plt.title('Magnitude PSNR Distribution')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Count')
    plt.savefig(HISTOGRAM_FILE)
    print(f"✓ Histogram saved to: {HISTOGRAM_FILE}")

if __name__ == '__main__':
    run_full_analysis()
# analyze_full_v3a.py
# Combined dataset analysis and spatial analysis

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd 
import scipy.io

from model_v3a import tiny_unet_small

# ========== CONFIGURATION ==========
EXPERIMENT_NAME = "V3c" 
PARENT_DIRECTORY = '/Users/leifefrancisco/Documents/Workspace/PI_CNN_V2/PI_CNN_V3a'

# Standardized Paths
BASE_PATH = '/Users/leifefrancisco/Documents/Workspace/PI_CNN_V2/PI_CNN_V3a'
DATA_PATH = '/Users/leifefrancisco/Documents/Workspace/PI_CNN_V2/PI_CNN_V3a/V3a_a_0_001/data'
RESULTS_PATH = '/Users/leifefrancisco/Documents/Workspace/PI_CNN_V2/PI_CNN_V3a/V3c_results/results_final_model'

# Inputs
MODEL_PATH = '/Users/leifefrancisco/Documents/Workspace/PI_CNN_V2/PI_CNN_V3a/V3b_results/final_model.pth'

# Outputs
CSV_REPORT_FILE = f"{RESULTS_PATH}/reconstruction_report.csv"
HISTOGRAM_FILE = f"{RESULTS_PATH}/psnr_histogram.png"
SPATIAL_FILE = f"{RESULTS_PATH}/spatial_analysis.png"
SAMPLE_VIS_FILE = f"{RESULTS_PATH}/sample_reconstructions.png"

ROOM_LENGTH = 3.0
ROOM_WIDTH = 3.0

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if torch.cuda.is_available(): DEVICE = torch.device("cuda")

# Ensure Output Dir Exists
Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)

# ========== DATASET CLASS ==========
class RoomAcousticDataset(Dataset):
    def __init__(self, data_dir=DATA_PATH, mode='val'):
        self.data_dir = Path(data_dir)
        self.file_list = sorted(list(self.data_dir.glob('room_acoustic_data_*.mat')))
        
        if len(self.file_list) == 0:
            print(f"ERROR: No .mat files found in {data_dir}")
            exit()
        
        split_idx = int(0.8 * len(self.file_list))
        if mode == 'train':
            self.file_list = self.file_list[:split_idx]
        elif mode == 'val':
            self.file_list = self.file_list[split_idx:]
        elif mode == 'all':
            pass  # Keep all files
    
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

def calculate_nmse(pred, target):
    mse = np.mean((pred - target) ** 2)
    target_energy = np.mean(target ** 2)
    if target_energy < 1e-10: return 0.0
    return 10 * np.log10(mse / target_energy)

# ========== ANALYSIS FUNCTIONS ==========
def check_mode_collapse(model, val_loader, device):
    """Check if model outputs vary across different inputs"""
    print("\n" + "="*60)
    print("CHECKING FOR MODE COLLAPSE")
    print("="*60)
    
    outputs_list = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 10: break
            outputs = model(batch['input'].to(device), batch['mask'].to(device))
            outputs_list.append(outputs[0, 0].cpu().numpy())
    
    stacked = np.stack(outputs_list)
    variance = np.var(stacked)
    mean_range = stacked.mean(axis=(1,2)).std()
    
    print(f"Output variance across samples: {variance:.6f}")
    print(f"Mean output range: {mean_range:.6f}")
    
    if variance < 0.001:
        print("⚠️  WARNING: Very low variance - possible mode collapse!")
        return True
    else:
        print("✓ Output variance looks healthy")
        return False

def run_reconstruction_analysis(model, val_loader, device):
    """Run reconstruction on all samples and collect metrics"""
    print("\n" + "="*60)
    print("RUNNING RECONSTRUCTION ANALYSIS")
    print("="*60)
    
    results = []
    all_predictions = []
    all_ground_truths = []
    all_inputs = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch['input'].to(device)
            masks = batch['mask'].to(device)
            gt_complex = batch['gt_complex'].numpy()[0]
            
            outputs = model(inputs, masks)
            
            pred_real = outputs[0, 0, :, :].cpu().numpy()
            pred_imag = outputs[0, 4, :, :].cpu().numpy()
            pred_complex = pred_real + 1j * pred_imag
            
            # Calculate metrics
            psnr_mag = calculate_psnr(np.abs(pred_complex), np.abs(gt_complex))
            psnr_real = calculate_psnr(pred_real, gt_complex.real)
            psnr_imag = calculate_psnr(pred_imag, gt_complex.imag)
            nmse = calculate_nmse(np.abs(pred_complex), np.abs(gt_complex))
            
            results.append({
                'filename': batch['filename'][0],
                'source_x': batch['source_x'].item(),
                'source_y': batch['source_y'].item(),
                'PSNR_Mag': psnr_mag,
                'PSNR_Real': psnr_real,
                'PSNR_Imag': psnr_imag,
                'NMSE_dB': nmse
            })
            
            # Store for visualization (first 5 samples)
            if len(all_predictions) < 5:
                all_predictions.append(pred_complex)
                all_ground_truths.append(gt_complex)
                all_inputs.append(batch['input'].numpy()[0])
            
            if (i+1) % 100 == 0: 
                print(f"  Processed {i+1} samples...")
    
    df = pd.DataFrame(results)
    print(f"\n✓ Analyzed {len(df)} samples")
    
    return df, all_predictions, all_ground_truths, all_inputs

def plot_histogram(df):
    """Plot PSNR histogram"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Magnitude PSNR
    axes[0].hist(df['PSNR_Mag'], bins=30, color='lightgreen', edgecolor='black')
    axes[0].set_title('Magnitude PSNR Distribution')
    axes[0].set_xlabel('PSNR (dB)')
    axes[0].set_ylabel('Count')
    axes[0].axvline(df['PSNR_Mag'].mean(), color='red', linestyle='--', label=f"Mean: {df['PSNR_Mag'].mean():.2f}")
    axes[0].legend()
    
    # Real PSNR
    axes[1].hist(df['PSNR_Real'], bins=30, color='lightblue', edgecolor='black')
    axes[1].set_title('Real Part PSNR Distribution')
    axes[1].set_xlabel('PSNR (dB)')
    axes[1].set_ylabel('Count')
    axes[1].axvline(df['PSNR_Real'].mean(), color='red', linestyle='--', label=f"Mean: {df['PSNR_Real'].mean():.2f}")
    axes[1].legend()
    
    # NMSE
    axes[2].hist(df['NMSE_dB'], bins=30, color='lightyellow', edgecolor='black')
    axes[2].set_title('NMSE Distribution')
    axes[2].set_xlabel('NMSE (dB)')
    axes[2].set_ylabel('Count')
    axes[2].axvline(df['NMSE_dB'].mean(), color='red', linestyle='--', label=f"Mean: {df['NMSE_dB'].mean():.2f}")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(HISTOGRAM_FILE)
    print(f"✓ Saved histogram to {HISTOGRAM_FILE}")
    plt.close()

def plot_spatial_analysis(df):
    """Plot spatial analysis - PSNR vs source position"""
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)

    # Plot 1: PSNR vs Source X
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(df['source_x'], df['PSNR_Mag'], alpha=0.5, color='royalblue', edgecolors='none')
    
    # Trend line
    z = np.polyfit(df['source_x'], df['PSNR_Mag'], 3)
    p = np.poly1d(z)
    x_range = np.linspace(df['source_x'].min(), df['source_x'].max(), 100)
    ax1.plot(x_range, p(x_range), "r--", linewidth=2, label='Trend')
    
    ax1.set_title(f'Performance vs. Source X Position')
    ax1.set_xlabel('Source X Coordinate (meters)')
    ax1.set_ylabel('Reconstruction Quality (PSNR dB)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: PSNR vs Source Y
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(df['source_y'], df['PSNR_Mag'], alpha=0.5, color='darkorange', edgecolors='none')
    
    z_y = np.polyfit(df['source_y'], df['PSNR_Mag'], 3)
    p_y = np.poly1d(z_y)
    y_range = np.linspace(df['source_y'].min(), df['source_y'].max(), 100)
    ax2.plot(y_range, p_y(y_range), "r--", linewidth=2, label='Trend')
    
    ax2.set_title(f'Performance vs. Source Y Position')
    ax2.set_xlabel('Source Y Coordinate (meters)')
    ax2.set_ylabel('Reconstruction Quality (PSNR dB)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Bird's Eye View (Spatial Heatmap)
    ax3 = fig.add_subplot(gs[1, :])
    sc = ax3.scatter(df['source_x'], df['source_y'], 
                     c=df['PSNR_Mag'], cmap='RdYlGn', 
                     s=80, edgecolors='black', alpha=0.8)
    
    plt.colorbar(sc, ax=ax3, label='PSNR (dB)')
    
    ax3.set_title("Bird's Eye View: Where does the model fail?\n(Red = Poor, Green = Good)")
    ax3.set_xlabel('Source X (meters)')
    ax3.set_ylabel('Source Y (meters)')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(SPATIAL_FILE)
    print(f"✓ Saved spatial analysis to {SPATIAL_FILE}")
    plt.close()

def plot_sample_reconstructions(predictions, ground_truths, inputs):
    """Plot a few sample reconstructions for visual inspection"""
    n_samples = len(predictions)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Input (sparse measurements)
        input_mag = np.sqrt(inputs[i][0]**2 + inputs[i][1]**2)
        im0 = axes[i, 0].imshow(input_mag, cmap='viridis', origin='lower')
        axes[i, 0].set_title(f'Sample {i+1}: Input (Sparse)')
        plt.colorbar(im0, ax=axes[i, 0])
        
        # Ground truth magnitude
        gt_mag = np.abs(ground_truths[i])
        im1 = axes[i, 1].imshow(gt_mag, cmap='viridis', origin='lower')
        axes[i, 1].set_title('Ground Truth')
        plt.colorbar(im1, ax=axes[i, 1])
        
        # Prediction magnitude
        pred_mag = np.abs(predictions[i])
        im2 = axes[i, 2].imshow(pred_mag, cmap='viridis', origin='lower')
        axes[i, 2].set_title('Prediction')
        plt.colorbar(im2, ax=axes[i, 2])
        
        # Error
        error = np.abs(pred_mag - gt_mag)
        im3 = axes[i, 3].imshow(error, cmap='hot', origin='lower')
        axes[i, 3].set_title(f'Error (PSNR: {calculate_psnr(pred_mag, gt_mag):.2f} dB)')
        plt.colorbar(im3, ax=axes[i, 3])
    
    plt.tight_layout()
    plt.savefig(SAMPLE_VIS_FILE)
    print(f"✓ Saved sample visualizations to {SAMPLE_VIS_FILE}")
    plt.close()

def print_summary_statistics(df):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nTotal samples analyzed: {len(df)}")
    print(f"\nPSNR (Magnitude):")
    print(f"  Mean:   {df['PSNR_Mag'].mean():.2f} dB")
    print(f"  Std:    {df['PSNR_Mag'].std():.2f} dB")
    print(f"  Min:    {df['PSNR_Mag'].min():.2f} dB")
    print(f"  Max:    {df['PSNR_Mag'].max():.2f} dB")
    print(f"\nNMSE:")
    print(f"  Mean:   {df['NMSE_dB'].mean():.2f} dB")
    print(f"  Std:    {df['NMSE_dB'].std():.2f} dB")
    print(f"\nSource Position Range:")
    print(f"  X: [{df['source_x'].min():.2f}, {df['source_x'].max():.2f}] meters")
    print(f"  Y: [{df['source_y'].min():.2f}, {df['source_y'].max():.2f}] meters")

# ========== MAIN ==========
def main():
    print("="*60)
    print(f"FULL ANALYSIS: {EXPERIMENT_NAME}")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = tiny_unet_small(in_channels=2, out_channels=8).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    
    model.eval()
    
    # Load dataset
    dataset = RoomAcousticDataset(DATA_PATH, mode='all')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"✓ Loaded {len(dataset)} samples")
    
    # Check for mode collapse
    collapsed = check_mode_collapse(model, loader, DEVICE)
    
    # Run reconstruction analysis
    df, predictions, ground_truths, inputs = run_reconstruction_analysis(model, loader, DEVICE)
    
    # Save CSV
    df.to_csv(CSV_REPORT_FILE, index=False)
    print(f"\n✓ Saved report to {CSV_REPORT_FILE}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_histogram(df)
    plot_spatial_analysis(df)
    plot_sample_reconstructions(predictions, ground_truths, inputs)
    
    # Print summary
    print_summary_statistics(df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()
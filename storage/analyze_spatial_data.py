# analyze_spatial_v2.py
# Extends the CSV analysis to plot Error vs. Room Position.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ========== CONFIGURATION ==========
VERSION_NUMBER = "V2a"
EXPERIMENT_NAME = "V2a_a_0_001" 
PARENT_DIRECTORY = f"PI_CNN_{VERSION_NUMBER}"

# Standardized Paths
BASE_PATH = f"./{PARENT_DIRECTORY}/{EXPERIMENT_NAME}"
DATA_PATH = f"{BASE_PATH}/data"
CSV_PATH = f"{DATA_PATH}/reconstruction_report.csv"
RESULTS_PATH = f"{BASE_PATH}/results"

ROOM_LENGTH = 3.0
ROOM_WIDTH = 3.0

def plot_spatial_analysis():
    # 1. Load Data
    csv_file = Path(CSV_PATH)
    if not csv_file.exists():
        print(f"Error: Could not find {CSV_PATH}")
        print("Please run analyze_dataset_v2.py first to generate the report!")
        return

    df = pd.read_csv(csv_file)
    print(f"Loaded metrics for {len(df)} scenarios.")

    # 2. Setup Plot Grid
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)

    # ==========================================
    # PLOT 1: THE GRAPH YOU ASKED FOR (X-Axis)
    # X = Source X Position, Y = PSNR (dB)
    # ==========================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Scatter plot
    ax1.scatter(df['source_x'], df['PSNR_Mag'], alpha=0.5, color='royalblue', edgecolors='none')
    
    # Add a trend line (Polynomial fit) to see the "average" performance across X
    z = np.polyfit(df['source_x'], df['PSNR_Mag'], 3) # 3rd order curve
    p = np.poly1d(z)
    x_range = np.linspace(0, ROOM_LENGTH, 100)
    ax1.plot(x_range, p(x_range), "r--", linewidth=2, label='Trend')

    ax1.set_title(f'Performance vs. Source X Position\n(Room Length: {ROOM_LENGTH}m)')
    ax1.set_xlabel('Source X Coordinate (meters)')
    ax1.set_ylabel('Reconstruction Quality (PSNR dB)')
    ax1.set_xlim(0, ROOM_LENGTH)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ==========================================
    # PLOT 2: Y-AXIS ANALYSIS
    # X = Source Y Position, Y = PSNR (dB)
    # ==========================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax2.scatter(df['source_y'], df['PSNR_Mag'], alpha=0.5, color='darkorange', edgecolors='none')
    
    # Trend line
    z_y = np.polyfit(df['source_y'], df['PSNR_Mag'], 3)
    p_y = np.poly1d(z_y)
    y_range = np.linspace(0, ROOM_WIDTH, 100)
    ax2.plot(y_range, p_y(y_range), "r--", linewidth=2, label='Trend')

    ax2.set_title(f'Performance vs. Source Y Position\n(Room Width: {ROOM_WIDTH}m)')
    ax2.set_xlabel('Source Y Coordinate (meters)')
    ax2.set_ylabel('Reconstruction Quality (PSNR dB)')
    ax2.set_xlim(0, ROOM_WIDTH)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # ==========================================
    # PLOT 3: SPATIAL HEATMAP (Bird's Eye View)
    # Shows "Dead Zones" in the room
    # ==========================================
    ax3 = fig.add_subplot(gs[1, :]) # Spans bottom row
    
    # We color the dots by PSNR (Red = Bad, Green = Good)
    sc = ax3.scatter(df['source_x'], df['source_y'], 
                     c=df['PSNR_Mag'], cmap='RdYlGn', 
                     s=80, edgecolors='black', alpha=0.8)
    
    plt.colorbar(sc, ax=ax3, label='PSNR (dB)')
    
    ax3.set_title('Bird\'s Eye View: Where does the model fail?\n(Red dots = Poor Reconstruction, Green dots = Good)')
    ax3.set_xlabel('Room Length (x)')
    ax3.set_ylabel('Room Width (y)')
    ax3.set_xlim(0, ROOM_LENGTH)
    ax3.set_ylim(0, ROOM_WIDTH)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.set_aspect('equal') # Make it look like the actual room shape

    # Save
    plt.tight_layout()
    save_path = Path(RESULTS_PATH)
    plt.savefig(save_path)
    print(f"\n✓ Saved graphs to {save_path}")
    plt.show()

if __name__ == '__main__':
    plot_spatial_analysis()
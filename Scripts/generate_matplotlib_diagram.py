import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Visual configuration
C_RAW = '#f9d0c4'    # Reddish
C_PROC = '#d4e157'   # Greenish
C_SCRIPT = '#b3e5fc' # Bluish
C_OUT = '#ce93d8'    # Purplish
C_MAN = '#ffcc80'    # Orange

fig, ax = plt.subplots(figsize=(15, 18))
ax.set_xlim(0, 15)
ax.set_ylim(-3, 16)
ax.axis('off')

def draw_box(x, y, width, height, text, bg_color, fontsize=10):
    # Shadow
    ax.add_patch(patches.Rectangle((x+0.05, y-0.05), width, height, facecolor='#dddddd', edgecolor='none', zorder=1))
    # Box
    ax.add_patch(patches.Rectangle((x, y), width, height, facecolor=bg_color, edgecolor='black', linewidth=1.5, zorder=2))
    # Text
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
            fontsize=fontsize, fontweight='bold', zorder=3, wrap=True)

def draw_arrow(x1, y1, x2, y2, color='black', lw=1.5, style="->"):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0), zorder=1)

# --- PIPELINE FLOW ---

# Layer 1: Source
draw_box(6, 14, 3, 1, '1. Raw Dataset\n(Electric / Motor-2 CSVs)', C_RAW, fontsize=11)

# Layer 2: Setup & Validation
draw_box(2, 12, 3, 1, 'Step 0: Setup PC\n(setup_pc.py)', C_SCRIPT)
draw_box(6, 12, 3, 1, 'Step 1: Validation\n(validate_dataset.py)', C_SCRIPT)
draw_arrow(7.5, 14, 7.5, 13) # Dataset to Validation
draw_arrow(7.5, 14, 3.5, 13, style='-', color='gray') # Dataset to Setup (implicit)

# Layer 3: THE MANIFEST (Central Hub)
draw_box(6, 10, 3, 1.2, 'Validated Manifest\n(Source of Truth)', C_MAN, fontsize=11)
draw_arrow(7.5, 12, 7.5, 11.2) # Validation to Manifest

# Layer 4: Parallel Processing (Denoising & Splitting)
draw_box(2, 8, 3, 1, 'Step 2: Denoising\n(denoise_all.py)', C_SCRIPT)
draw_box(10, 8, 3, 1, 'Step 3: Split Gen\n(generate_splits_v2.py)', C_SCRIPT)

# Connections to Manifest (The Gatekeeper)
draw_arrow(6, 10.6, 5, 10.6, style='-') # From manifest side
draw_arrow(5, 10.6, 3.5, 9) # To Denoising
draw_arrow(9, 10.6, 10, 10.6, style='-') # From manifest side
draw_arrow(10, 10.6, 11.5, 9) # To Splitting

# Connection from Dataset to Denoising
draw_arrow(6, 14.5, 1, 14.5, style='-', color='gray') # Dataset to far left
draw_arrow(1, 14.5, 1, 8.5, style='-', color='gray')
draw_arrow(1, 8.5, 2, 8.5, color='gray') # Into Denoising

# Layer 5: Processing Outputs
draw_box(2, 6, 3, 1, 'Denoised Hub\n(Filtered Signals)', C_PROC)
draw_box(10, 6, 3, 1, 'Splits CSV\n(v2 Speed-Strat)', C_MAN)
draw_arrow(3.5, 8, 3.5, 7) # Denoising to Hub
draw_arrow(11.5, 8, 11.5, 7) # Splitting to Splits CSV

# Layer 6: Feature Extraction (Step 4)
# Horizontal Precompute Bus
draw_box(1, 3.5, 13, 1.5, 'Step 4: Feature Precomputation Hub\n(STFT, Envelope, DWT)', C_SCRIPT, fontsize=12)

# Inputs to Precompute Hub
draw_arrow(3.5, 6, 3.5, 5, lw=2, color='#2ecc71') # Denoised Hub -> Hub
draw_arrow(11.5, 6, 11.5, 5, lw=2, color='#f39c12') # Splits CSV -> Hub

# Layer 7: Artifacts (Branches)
draw_box(1.5, 1.5, 3, 1, '4a. STFT\n(.pt, .png)', C_OUT)
draw_box(6, 1.5, 3, 1, '4b. Envelope\n(.pt)', C_OUT)
draw_box(10.5, 1.5, 3, 1, '4c. DWT\n(.pt, .png)', C_OUT)

draw_arrow(3, 3.5, 3, 2.5)
draw_arrow(7.5, 3.5, 7.5, 2.5)
draw_arrow(12, 3.5, 12, 2.5)

# Layer 8: Individual Training Stages (Step 5)
draw_box(1.5, -0.5, 3, 1.2, 'Step 5a: Train STFT CNN\n(train_stft_cnn_*.py)', C_SCRIPT)
draw_box(6, -0.5, 3, 1.2, 'Step 5b: Train Env CNN\n(train_envelope_cnn_*.py)', C_SCRIPT)
draw_box(10.5, -0.5, 3, 1.2, 'Step 5c: Train DWT CNN\n(train_dwt_cnn.py)', C_SCRIPT)

draw_arrow(3, 1.5, 3, 0.7)
draw_arrow(7.5, 1.5, 7.5, 0.7)
draw_arrow(12, 1.5, 12, 0.7)

# Layer 9: Final Deliverables (Step 6)
draw_box(4.5, -2.5, 6, 1.2, 'Step 6: Final Model Outputs\n(Metrics, CMs, Best Models, Diagnostics)', C_OUT, fontsize=11)
draw_arrow(3, -0.5, 7.5, -1.3)
draw_arrow(7.5, -0.5, 7.5, -1.3)
draw_arrow(12, -0.5, 7.5, -1.3)

# Legend
draw_box(12, 14.5, 2.5, 0.5, 'Raw Data', C_RAW)
draw_box(12, 13.8, 2.5, 0.5, 'Processed Data', C_PROC)
draw_box(12, 13.1, 2.5, 0.5, 'Script / Python', C_SCRIPT)
draw_box(12, 12.4, 2.5, 0.5, 'Manifest / Metadata', C_MAN)
draw_box(12, 11.7, 2.5, 0.5, 'Final Artifacts', C_OUT)

plt.title('Motor Fault Diagnosis Pipeline Architecture (STFT / Envelope / DWT)', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()

os.makedirs('ProjectShare', exist_ok=True)
plt.savefig('ProjectShare/pipeline_diagram_300dpi.jpg', dpi=300, bbox_inches='tight', format='jpg')
plt.savefig('ProjectShare/pipeline_diagram.svg', bbox_inches='tight', format='svg')
print("Updated diagrams with Envelope and individual training steps generated successfully in ProjectShare/")

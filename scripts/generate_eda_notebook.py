"""Script to build notebooks/01_african_data_exploration.ipynb using nbformat."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Markdown Title
cells.append(nbf.v4.new_markdown_cell("""# 🛰️ Thermal-Edge-Fire: Pan-African Exploratory Data Analysis (EDA)
### Onboard Thermal/IR Edge-Computing for Wildfire Detection on a 6U CubeSat

This notebook provides an interactive **Exploratory Data Analysis (EDA)** of satellite thermal imagery over the African continent, focusing on:
1. **Geographic Distribution**: North Africa / Mediterranean Basin (Tunisia / Maghreb) vs. Sub-Saharan Africa.
2. **Thermal Channel Visualizations**: Landsat-8 TIRS Band 10 (10.9 µm LWIR) and Band 11 (12.0 µm LWIR).
3. **Physical Fire Separability**: Radiant intensity elevation of active fire fronts above hot desert/savanna backgrounds.
4. **Gate 1 Validation Quality Gate**: Pre-treatment sensor telemetry verification (Ground Mode vs. Flight PSF Halo Mode).
"""))

# Cell 2: Imports & Environment Setup
cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from fireedge.config import load_config, resolve_path, get_reports_dir
from fireedge.io import read_image, read_mask, index_patches
from fireedge.validation import GroundGateValidator, FlightGateValidator, ValidatorConfig

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
%matplotlib inline
print("Environment initialized successfully.")
"""))

# Cell 3: Markdown - Inventory Overview
cells.append(nbf.v4.new_markdown_cell("""## 1. African Dataset Inventory & Geolocation
We load the curated African inventory (`reports/africa_inventory.csv`) derived from Pereira et al. (ActiveFire).
Each scene has been inverse-projected from UTM projection coordinates to geographic WGS84 (${\\rm Lat}, {\\rm Lon}$).
"""))

# Cell 4: Load Inventory & Display Summary
cells.append(nbf.v4.new_code_cell("""reports_dir = PROJECT_ROOT / "reports"
inv_path = reports_dir / "africa_inventory.csv"

if not inv_path.exists():
    print(f"Generating inventory from ActiveFire root...")
    cfg = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    data_root = resolve_path(cfg["data"]["activefire_root"], PROJECT_ROOT)
    df_all = index_patches(data_root)
    df_africa = df_all[df_all["region"].isin(["NORTH_AFRICA_MED", "SUB_SAHARAN_AFRICA"])].copy()
    df_africa.to_csv(inv_path, index=False)
else:
    df_africa = pd.read_csv(inv_path)

print(f"Total African Scenes: {len(df_africa)}")
print("\\nRegional Distribution:")
display(df_africa["region"].value_counts().to_frame("Count"))
"""))

# Cell 5: Geographic Map Plot
cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot of African scenes by region
colors = {"NORTH_AFRICA_MED": "#E63946", "SUB_SAHARAN_AFRICA": "#2A9D8F"}
for region, group in df_africa.groupby("region"):
    ax.scatter(group["center_lon"], group["center_lat"], 
               c=colors.get(region, "#457B9D"), label=region, s=70, alpha=0.85, edgecolors='black', linewidth=0.5)

# Highlight Tunisia / North Africa target area
ax.axhspan(27, 38, xmin=0.3, xmax=0.55, color='orange', alpha=0.15, label='Tunisian / Maghreb Target Zone')

ax.set_title("Geographic Distribution of African Satellite Wildfire Patches", fontsize=14, weight='bold')
ax.set_xlabel("Longitude (°E)", fontsize=12)
ax.set_ylabel("Latitude (°N)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=11)

plt.tight_layout()
plt.show()
"""))

# Cell 6: Markdown - Visualizing Thermal Bands
cells.append(nbf.v4.new_markdown_cell("""## 2. Multi-Channel Thermal Inspection
Here we visualize a representative African wildfire patch across:
- **Band 10 (10.9 µm LWIR)**: Primary thermal channel for microbolometer CubeSat payload.
- **Band 11 (12.0 µm LWIR)**: Secondary thermal channel for ground-truth cross-confirmation.
- **Thermal Difference $(B10 - B11)$**: Differential absorption highlighting smoke and active fire emissivity.
- **Voting Fire Mask**: Human & multi-algorithm consensus active fire ground truth.
"""))

# Cell 7: Load and Plot Multi-Channel Patch
cells.append(nbf.v4.new_code_cell("""# Pick a sample patch with active fire
fire_patches = df_africa[df_africa["has_mask"]]
sample_row = fire_patches.iloc[10]

img = read_image(sample_row["image_path"])
mask = read_mask(sample_row["mask_path"], shape=img.shape[1:])

b10 = img[8] # Channel index 8 = Band 10
b11 = img[9] # Channel index 9 = Band 11
b10_sub_b11 = b10.astype(float) - b11.astype(float)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

im0 = axes[0].imshow(b10, cmap="inferno")
axes[0].set_title(f"Band 10 (LWIR 10.9 µm)\\nRange: [{b10.min()}, {b10.max()}] DN", fontsize=11)
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(b11, cmap="inferno")
axes[1].set_title(f"Band 11 (LWIR 12.0 µm)\\nRange: [{b11.min()}, {b11.max()}] DN", fontsize=11)
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(b10_sub_b11, cmap="coolwarm")
axes[2].set_title(f"Dual-Band Diff (B10 - B11)\\nAnomalies & Plumes", fontsize=11)
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

im3 = axes[3].imshow(mask, cmap="Reds", interpolation="nearest")
axes[3].set_title(f"Active Fire Mask\\nFire Pixels: {int((mask > 0).sum())}", fontsize=11)
plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

for ax in axes:
    ax.axis("off")

plt.suptitle(f"Scene: {sample_row['stem']} ({sample_row['region']})", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()
"""))

# Cell 8: Markdown - Physical Separability Analysis
cells.append(nbf.v4.new_markdown_cell("""## 3. Physical Thermal Separability Analysis
In John McDonald's feedback (§6.3), a critical question is whether thermal-only LWIR data is sufficient to detect wildfires without false alarms on hot bare ground.
Below, we evaluate the Digital Number (DN) distribution of **active fire pixels** versus **background land pixels** across all African scenes.
"""))

# Cell 9: Plot Separability Histograms & ROC Curve
cells.append(nbf.v4.new_code_cell("""from fireedge.exploration import band_separability_report

channel_map = {"B6 (SWIR 1)": 5, "B7 (SWIR 2)": 6, "B10 (LWIR 1)": 8, "B11 (LWIR 2)": 9}
fire_samples = {b: [] for b in channel_map}
bg_samples = {b: [] for b in channel_map}

for _, row in df_africa[df_africa["has_mask"]].iterrows():
    p_img = read_image(row["image_path"])
    p_mask = read_mask(row["mask_path"], shape=p_img.shape[1:]) > 0
    bg_m = ~p_mask & (p_img[8] > 0)
    for b_name, c_idx in channel_map.items():
        if c_idx < p_img.shape[0]:
            f_pix = p_img[c_idx][p_mask]
            b_pix = p_img[c_idx][bg_m]
            if len(f_pix) > 0:
                fire_samples[b_name].append(f_pix)
            if len(b_pix) > 0:
                bg_samples[b_name].append(b_pix[::50]) # Subsample background

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Band 10 Distribution
b10_f = np.concatenate(fire_samples["B10 (LWIR 1)"])
b10_bg = np.concatenate(bg_samples["B10 (LWIR 1)"])

axes[0].hist(b10_bg, bins=60, density=True, alpha=0.6, color="navy", label=f"Background (median={np.median(b10_bg):.0f} DN)")
axes[0].hist(b10_f, bins=60, density=True, alpha=0.7, color="red", label=f"Fire Pixels (median={np.median(b10_f):.0f} DN)")
axes[0].set_title("Band 10 (LWIR 10.9 µm) Pixel Density on African Terrains", fontsize=12, weight='bold')
axes[0].set_xlabel("Digital Number (DN)")
axes[0].set_ylabel("Probability Density")
axes[0].legend()

# Band 11 Distribution
b11_f = np.concatenate(fire_samples["B11 (LWIR 2)"])
b11_bg = np.concatenate(bg_samples["B11 (LWIR 2)"])

axes[1].hist(b11_bg, bins=60, density=True, alpha=0.6, color="navy", label=f"Background (median={np.median(b11_bg):.0f} DN)")
axes[1].hist(b11_f, bins=60, density=True, alpha=0.7, color="darkorange", label=f"Fire Pixels (median={np.median(b11_f):.0f} DN)")
axes[1].set_title("Band 11 (LWIR 12.0 µm) Pixel Density on African Terrains", fontsize=12, weight='bold')
axes[1].set_xlabel("Digital Number (DN)")
axes[1].set_ylabel("Probability Density")
axes[1].legend()

plt.tight_layout()
plt.show()

# Print ROC-AUC Metrics
samples_dict = {
    "B10": (b10_f, b10_bg),
    "B11": (b11_f, b11_bg),
    "B7": (np.concatenate(fire_samples["B7 (SWIR 2)"]), np.concatenate(bg_samples["B7 (SWIR 2)"]))
}
report_df = band_separability_report(samples_dict)
print("=== African Separability Summary ===")
display(report_df)
"""))

# Cell 10: Markdown - Gate 1 Quality Gate in Action
cells.append(nbf.v4.new_markdown_cell("""## 4. Gate 1 Validation: Ground Mode vs. Flight Mode
In this section, we run the **Dual-Gate Input Validator** directly on a raw frame:
1. **Ground Mode (`GroundGateValidator`)**: Cross-checks hot pixels between Band 10 and Band 11.
2. **Flight Mode (`FlightGateValidator`)**: Analyzes single-band Point Spread Function (PSF) Spatial Halo Ratio to differentiate cosmic ray radiation (SEU) hits from real sub-pixel optical fires.
"""))

# Cell 11: Execute Validator Live in Notebook
cells.append(nbf.v4.new_code_cell("""# Calibrate a standard 16-bit configuration
b10_cfg = ValidatorConfig(bit_depth=16, valid_min=15000, valid_max=50000, noise_flag=5.0, noise_reject=15.0)
b11_cfg = ValidatorConfig(bit_depth=16, valid_min=15000, valid_max=45000, noise_flag=5.0, noise_reject=15.0)

# Instantiate validators
ground_val = GroundGateValidator([b10_cfg, b11_cfg])
flight_val = FlightGateValidator(b10_cfg, halo_threshold=0.15)

# Validate sample
dual_band_img = img[[8, 9]] # B10, B11
single_band_img = img[[8]]   # B10 only

rep_ground = ground_val.validate(dual_band_img)
rep_flight = flight_val.validate(single_band_img)

print("=== Gate 1 Validation Verdicts ===")
print(f"Ground Mode Verdict: {rep_ground.status.value}")
print(f"  Reasons: {rep_ground.reasons}")
print(f"  B10 confirmed anomalies: {rep_ground.metrics.get('ch0.hot_confirmed_real', 0)}")
print()
print(f"Flight Mode Verdict: {rep_flight.status.value}")
print(f"  Reasons: {rep_flight.reasons}")
print(f"  Confirmed Optical Hotspots: {rep_flight.metrics.get('ch0.hot_confirmed_optical', 0)}")
print(f"  Radiation SEU Transients Filtered: {rep_flight.metrics.get('ch0.seu_transients', 0)}")
"""))

# Cell 12: Markdown - Why Data Is Not Blindly Cleaned
cells.append(nbf.v4.new_markdown_cell("""## 5. Architectural Note: Why Don't We "Clean" Data at the Validation Gate?
A common question in satellite edge AI is: *Why doesn't the validation gate clean/smooth the image?*

1. **Protecting Wildfire Recall (§6.3 John McDonald rule)**:
   - A real nascent wildfire occupies only **1 to 5 pixels** in an image.
   - Traditional filtering (e.g., median blur, gaussian despiking) **erases** or dilutes sub-pixel fires, causing fatal false-negatives (missed fires).
   - Gate 1 uses a 3-level verdict: `PASS`, `PASS_WITH_FLAGS`, and `REJECT`. If an anomaly is present, it is recorded, but pixel values remain intact for the neural network.
2. **Selective SEU Repair in Flight Mode**:
   - Only when a pixel is proven to be a **radiation hit** (zero optical halo blur, $\\text{Halo Ratio} < 0.15$) does the optional `hot_pixel_policy="repair"` replace the pixel with the local median.
3. **Stage Separation (QA vs. Pre-processing)**:
   - **Gate 1** is an **Auditor** (Quality Assurance).
   - Actual calibration, Non-Uniformity Correction (NUC), and dynamic range scaling occur downstream in **Stage 2 (Pre-Processing)**.
"""))

nb.cells = cells

out_path = Path("notebooks/01_african_data_exploration.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Generated {out_path} with {len(cells)} cells.")

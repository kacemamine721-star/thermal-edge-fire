"""Build the self-contained Colab notebook for the Africa Tiny U-Net v2 run."""
from pathlib import Path

import nbformat as nbf


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"name": "03_train_africa_balanced_tiny_unet.ipynb", "provenance": []},
}

nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Africa Tiny U-Net v2 — balanced strict-consensus training\n\n"
        "This Colab notebook downloads a bounded Africa subset, creates strict voting masks, "
        "adds only safe non-fire patches, runs Ground Gate 1, trains a lightweight B10 Tiny U-Net, "
        "and exports the checkpoint, ONNX model, metrics, and figures."
    ),
    code(r'''
!df -h /content
!pip -q install gdown rasterio tqdm pandas numpy matplotlib scipy onnx

from google.colab import drive
drive.mount("/content/drive")

PROJECT_ZIP = "/content/drive/MyDrive/thermal-edge-fire.zip"  # Change only if your ZIP is elsewhere.
assert __import__("os").path.exists(PROJECT_ZIP), f"Project ZIP not found: {PROJECT_ZIP}"

!rm -rf /content/thermal-edge-fire
!unzip -q "$PROJECT_ZIP" -d /content
%cd /content/thermal-edge-fire
print("Project extracted.")
'''),
    code(r'''
from pathlib import Path

DATA_DIR = Path("/content")
AFRICA_ARCHIVE = DATA_DIR / "Africa.zip"

AFRICA_URL = "https://drive.google.com/file/d/1Ng3JwsjJPApshk8lJdGcsNHI52NaEMDX/view"

if not AFRICA_ARCHIVE.exists():
    !gdown --fuzzy "$AFRICA_URL" -O "$AFRICA_ARCHIVE"

assert AFRICA_ARCHIVE.exists(), "Africa.zip download failed."
print("Archive size (GB):", round(AFRICA_ARCHIVE.stat().st_size / 1024**3, 2))
'''),
    code(r'''
# ================================================================
# STORAGE-SAFE SUBSET EXTRACTION
# Keeps 3,000 image patches, their three algorithmic masks only.
# ================================================================
import random, re, shutil, tempfile, zipfile
from pathlib import Path
from tqdm.auto import tqdm

SEED = 42
MAX_PATCHES = 3000
MAX_PATCHES_PER_SCENE = 8
ALGORITHMS = ("Kumar-Roy", "Murphy", "Schroeder")

PREPARED_ROOT = Path("/content/africa_prepared")
PATCH_DIR = PREPARED_ROOT / "patches"
ALGORITHM_MASK_DIR = PREPARED_ROOT / "masks" / "algorithmic"

# Rebuild only when changing the selected subset.
!rm -rf /content/africa_prepared
PATCH_DIR.mkdir(parents=True, exist_ok=True)
ALGORITHM_MASK_DIR.mkdir(parents=True, exist_ok=True)

def scene_id(member):
    return re.sub(r"(_masks)?\.zip$", "", Path(member).name)

def canonical_patch(mask_name):
    return re.sub(r"_(Kumar-Roy|Murphy|Schroeder)_p", "_p", Path(mask_name).name)

def open_nested(archive_path, member):
    buffer = tempfile.SpooledTemporaryFile(max_size=256 * 1024 * 1024)
    with zipfile.ZipFile(archive_path, "r") as archive:
        with archive.open(member) as src:
            shutil.copyfileobj(src, buffer)
    buffer.seek(0)
    return buffer, zipfile.ZipFile(buffer, "r")

with zipfile.ZipFile(AFRICA_ARCHIVE, "r") as archive:
    members = archive.namelist()

image_members = [m for m in members if re.fullmatch(r"z\d+\.zip", Path(m).name)]
mask_members = [m for m in members if re.fullmatch(r"z\d+_masks\.zip", Path(m).name)]
print("Image scene archives:", len(image_members), "Mask scene archives:", len(mask_members))

rng = random.Random(SEED)
rng.shuffle(image_members)
selected_patches, selected_scenes = set(), set()

for member in tqdm(image_members, desc="Extracting image patches"):
    if len(selected_patches) >= MAX_PATCHES:
        break
    buffer, nested = open_nested(AFRICA_ARCHIVE, member)
    try:
        infos = [i for i in nested.infolist() if not i.is_dir() and i.filename.lower().endswith(".tif")]
        rng.shuffle(infos)
        for info in infos[:MAX_PATCHES_PER_SCENE]:
            if len(selected_patches) >= MAX_PATCHES:
                break
            name = Path(info.filename).name
            with nested.open(info) as src, open(PATCH_DIR / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            selected_patches.add(name)
            selected_scenes.add(scene_id(member))
    finally:
        nested.close(); buffer.close()

mask_by_scene = {scene_id(m): m for m in mask_members}
saved_masks = 0
for sid in tqdm(sorted(selected_scenes), desc="Extracting matching masks"):
    member = mask_by_scene.get(sid)
    if member is None:
        continue
    buffer, nested = open_nested(AFRICA_ARCHIVE, member)
    try:
        for info in nested.infolist():
            name = Path(info.filename).name
            if info.is_dir() or not name.lower().endswith(".tif"):
                continue
            if not any(f"_{algorithm}_p" in name for algorithm in ALGORITHMS):
                continue
            if canonical_patch(name) not in selected_patches:
                continue
            with nested.open(info) as src, open(ALGORITHM_MASK_DIR / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            saved_masks += 1
    finally:
        nested.close(); buffer.close()

print("Images:", len(selected_patches), "algorithmic masks:", saved_masks)
!df -h /content
'''),
    code(r'''
# ================================================================
# STRICT VOTING + DATA EXPLORATION + SAFE NON-FIRE PATCHES
# A patch with exactly one label algorithm is uncertain and excluded.
# A patch with zero label algorithms is a non-fire candidate under the
# ActiveFire dataset convention (only fire masks are stored).
# ================================================================
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import rasterio
from tqdm.auto import tqdm

VOTING_DIR = PREPARED_ROOT / "masks" / "voting_strict"
NONFIRE_DIR = PREPARED_ROOT / "masks" / "nonfire_generated"
VOTING_DIR.mkdir(parents=True, exist_ok=True)
NONFIRE_DIR.mkdir(parents=True, exist_ok=True)

def algorithm_name(stem):
    low = stem.lower()
    if "kumar-roy" in low or "kumar_roy" in low: return "kumar_roy"
    if "murphy" in low: return "murphy"
    if "schroeder" in low: return "schroeder"
    return None

def image_stem(mask_stem):
    return re.sub(r"_(Kumar[-_]?Roy|Murphy|Schroeder)_p", "_p", mask_stem, flags=re.I)

image_files = {p.stem: p for p in PATCH_DIR.glob("*.tif")}
masks_by_image = defaultdict(dict)
for mask_file in ALGORITHM_MASK_DIR.glob("*.tif"):
    algorithm = algorithm_name(mask_file.stem)
    target = image_stem(mask_file.stem)
    if algorithm and target in image_files:
        masks_by_image[target][algorithm] = mask_file

strict_records, coverage_records = [], []
for stem, image_file in tqdm(image_files.items(), desc="Creating voting masks"):
    sources = masks_by_image.get(stem, {})
    coverage_records.append({"stem": stem, "image_path": str(image_file), "n_label_algorithms": len(sources)})
    if len(sources) < 2:
        continue
    masks, profile = [], None
    for algorithm in ("kumar_roy", "murphy", "schroeder"):
        path = sources.get(algorithm)
        if path is None:
            continue
        with rasterio.open(path) as src:
            masks.append((src.read(1) > 0).astype(np.uint8))
            if profile is None:
                profile = src.profile.copy()
    vote = (np.sum(masks, axis=0) >= 2).astype(np.uint8)
    profile.pop("blockxsize", None); profile.pop("blockysize", None); profile.pop("tiled", None)
    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="lzw")
    out = VOTING_DIR / f"{stem}_voting.tif"
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(vote, 1)
    match = re.search(r"LC08_[A-Z0-9]+_(\d{3})(\d{3})_", stem)
    strict_records.append({"stem": stem, "image_path": str(image_file), "mask_path": str(out),
                           "n_label_algorithms": len(masks), "fire_pixels": int(vote.sum()),
                           "path": int(match.group(1)) if match else None,
                           "row": int(match.group(2)) if match else None})

strict_inventory = pd.DataFrame(strict_records)
coverage = pd.DataFrame(coverage_records)
print("Label-source coverage:")
display(coverage.n_label_algorithms.value_counts().sort_index())
print("Strict pairs:", len(strict_inventory), "strict zero masks:", int((strict_inventory.fire_pixels == 0).sum()))

positive = strict_inventory[strict_inventory.fire_pixels > 0].copy()
strict_zero = strict_inventory[strict_inventory.fire_pixels == 0].copy()
candidates = coverage[coverage.n_label_algorithms == 0].copy()
N_NEGATIVES = min(1200, len(candidates))
if N_NEGATIVES == 0:
    print("WARNING: This Africa archive contains only fire-candidate patches. "
          "No genuine image-level negatives are available; training will remain strict-only. "
          "Do not convert one-algorithm patches into negative labels.")
    negatives = candidates.iloc[0:0].copy()
else:
    negatives = candidates.sample(N_NEGATIVES, random_state=SEED)

negative_records = []
for row in tqdm(negatives.itertuples(index=False), total=len(negatives), desc="Writing zero masks"):
    source, out = Path(row.image_path), NONFIRE_DIR / f"{row.stem}_nonfire.tif"
    with rasterio.open(source) as src:
        profile = src.profile.copy()
        profile.pop("blockxsize", None); profile.pop("blockysize", None); profile.pop("tiled", None)
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="lzw")
        zeros = np.zeros((src.height, src.width), dtype=np.uint8)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(zeros, 1)
    match = re.search(r"LC08_[A-Z0-9]+_(\d{3})(\d{3})_", row.stem)
    negative_records.append({"stem": row.stem, "image_path": str(source), "mask_path": str(out),
                             "n_label_algorithms": 0, "fire_pixels": 0,
                             "path": int(match.group(1)) if match else None,
                             "row": int(match.group(2)) if match else None,
                             "sample_type": "nonfire"})

positive["sample_type"] = "strict_fire"
strict_zero["sample_type"] = "strict_zero"
df_all = pd.concat([positive, strict_zero, pd.DataFrame(negative_records)], ignore_index=True)
BALANCED_INVENTORY = PREPARED_ROOT / "training_inventory_balanced_v2.csv"
df_all.to_csv(BALANCED_INVENTORY, index=False)

print("Balanced training inventory:")
display(df_all.sample_type.value_counts())
print("Zero-fire patches:", int((df_all.fire_pixels == 0).sum()), "/", len(df_all))
display(df_all.fire_pixels.describe(percentiles=[.1, .25, .5, .75, .9, .99]))
'''),
    code(r'''
# ================================================================
# PROJECT WORKFLOW + GATE 1 GROUND CURATION
# Gate 1 uses B10 and B11; training itself remains B10-only.
# ================================================================
import sys, random
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/content/thermal-edge-fire")
OUTPUT_DIR = PROJECT_ROOT / "reports" / "training_africa_tiny_unet_balanced_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fireedge.io import read_image, read_mask
from fireedge.data.dataset import ActiveFireDataset
from fireedge.validation import GroundGateValidator, Status, ValidatorConfig, calibrate_from_frames

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda": print("GPU:", torch.cuda.get_device_name(0))

assert df_all.image_path.map(lambda p: Path(p).exists()).all()
assert df_all.mask_path.map(lambda p: Path(p).exists()).all()

# Calibrate Gate 1 thresholds from a small representative B10/B11 sample.
calibration_df = df_all.sample(min(64, len(df_all)), random_state=SEED)
frames = [read_image(path, channels=(8, 9)).astype(np.float32) for path in calibration_df.image_path]
cfg_b10 = calibrate_from_frames((frame[0] for frame in frames), ValidatorConfig(hot_pixel_policy="flag"))
cfg_b11 = calibrate_from_frames((frame[1] for frame in frames), ValidatorConfig(hot_pixel_policy="flag"))
ground_gate = GroundGateValidator([cfg_b10, cfg_b11])

gate_rows = []
for row in tqdm(df_all.itertuples(), total=len(df_all), desc="Gate 1 ground curation"):
    report = ground_gate.validate(read_image(row.image_path, channels=(8, 9)).astype(np.float32))
    gate_rows.append({"stem": row.stem, "gate_status": report.status.value, "gate_reasons": "|".join(report.reasons)})

gate_report = pd.DataFrame(gate_rows)
gate_report.to_csv(OUTPUT_DIR / "gate1_ground_audit.csv", index=False)
display(gate_report.gate_status.value_counts())

rejected = set(gate_report.loc[gate_report.gate_status == Status.REJECT.value, "stem"])
df_all = df_all[~df_all.stem.isin(rejected)].copy()
assert len(df_all) > 100, "Gate 1 rejected too much data; inspect gate1_ground_audit.csv."
print("Pairs retained after Gate 1:", len(df_all))
'''),
    code(r'''
# ================================================================
# SCENE-SAFE SPLIT + FINAL BALANCE AUDIT
# ================================================================
df_all["parent_scene"] = df_all.stem.str.replace(r"_p\d+$", "", regex=True)
rng = np.random.default_rng(SEED)
scenes = df_all.parent_scene.drop_duplicates().to_numpy()
rng.shuffle(scenes)
n_test, n_val = max(1, round(.15 * len(scenes))), max(1, round(.15 * len(scenes)))
test_scenes = set(scenes[:n_test])
val_scenes = set(scenes[n_test:n_test + n_val])
train_scenes = set(scenes[n_test + n_val:])
train_df = df_all[df_all.parent_scene.isin(train_scenes)].copy()
val_df = df_all[df_all.parent_scene.isin(val_scenes)].copy()
test_df = df_all[df_all.parent_scene.isin(test_scenes)].copy()

for name, part in [("train", train_df), ("validation", val_df), ("test", test_df)]:
    assert not part.empty
    part.to_csv(OUTPUT_DIR / f"{name}_inventory.csv", index=False)
    print(f"{name:10s} patches={len(part):,} scenes={part.parent_scene.nunique():,} "
          f"zero_masks={(part.fire_pixels == 0).sum():,} fire_pixels={part.fire_pixels.sum():,}")
    display(part.sample_type.value_counts().rename(name))

assert not (set(train_df.parent_scene) & set(val_df.parent_scene))
assert not (set(train_df.parent_scene) & set(test_df.parent_scene))
assert not (set(val_df.parent_scene) & set(test_df.parent_scene))
'''),
    code(r'''
# ================================================================
# B10 DATA LOADERS + 121K-PARAMETER TINY U-NET
# ================================================================
BATCH_SIZE = 8 if device.type == "cuda" else 2
NUM_WORKERS = 2 if device.type == "cuda" else 0

train_ds = ActiveFireDataset(train_df, channels=(8,), norm_method="fixed", augment=True, random_seed=SEED)
val_ds = ActiveFireDataset(val_df, channels=(8,), norm_method="fixed", augment=False)
test_ds = ActiveFireDataset(test_df, channels=(8,), norm_method="fixed", augment=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
                                   nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))
    def forward(self, x): return self.block(x)

class TinyUNet(nn.Module):
    def __init__(self, base=8):
        super().__init__()
        self.e1, self.e2, self.e3 = ConvBlock(1,base), ConvBlock(base,base*2), ConvBlock(base*2,base*4)
        self.pool, self.mid = nn.MaxPool2d(2), ConvBlock(base*4,base*8)
        self.u3, self.d3 = nn.ConvTranspose2d(base*8,base*4,2,2), ConvBlock(base*8,base*4)
        self.u2, self.d2 = nn.ConvTranspose2d(base*4,base*2,2,2), ConvBlock(base*4,base*2)
        self.u1, self.d1 = nn.ConvTranspose2d(base*2,base,2,2), ConvBlock(base*2,base)
        self.out = nn.Conv2d(base,1,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); m=self.mid(self.pool(e3))
        d3=self.d3(torch.cat([self.u3(m),e3],1)); d2=self.d2(torch.cat([self.u2(d3),e2],1)); d1=self.d1(torch.cat([self.u1(d2),e1],1))
        return self.out(d1)

model = TinyUNet(base=8).to(device)
print("Parameters:", sum(p.numel() for p in model.parameters()))
x, y = next(iter(train_loader)); print("Batch:", x.shape, y.shape, "range:", (x.min().item(), x.max().item()))
'''),
    code(r'''
# ================================================================
# LOSS, TRAINING, VALIDATION, AND EARLY STOPPING
# ================================================================
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=.3, beta=.7, gamma=.75, eps=1e-6):
        super().__init__(); self.alpha, self.beta, self.gamma, self.eps = alpha, beta, gamma, eps
    def forward(self, logits, targets):
        p = torch.sigmoid(logits); d = (1,2,3)
        tp=(p*targets).sum(d); fp=(p*(1-targets)).sum(d); fn=((1-p)*targets).sum(d)
        tv=(tp+self.eps)/(tp+self.alpha*fp+self.beta*fn+self.eps)
        return ((1-tv)**self.gamma).mean()

bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([25.], device=device))
tversky = FocalTverskyLoss()
def loss_fn(logits, targets): return .3*bce(logits,targets) + .7*tversky(logits,targets)

@torch.no_grad()
def evaluate(loader, threshold=.2):
    model.eval(); tp=fp=fn=0; losses=[]
    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        logits=model(images); losses.append(loss_fn(logits,masks).item())
        pred=torch.sigmoid(logits)>=threshold; target=masks.bool()
        tp+=(pred&target).sum().item(); fp+=(pred&~target).sum().item(); fn+=(~pred&target).sum().item()
    recall=tp/(tp+fn+1e-9); precision=tp/(tp+fp+1e-9); iou=tp/(tp+fp+fn+1e-9); dice=2*tp/(2*tp+fp+fn+1e-9)
    return {"loss":float(np.mean(losses)),"recall":recall,"precision":precision,"iou":iou,"dice":dice,"tp":tp,"fp":fp,"fn":fn}

optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
EPOCHS, PATIENCE = 30, 6
history=[]; best_dice=-1.; stale=0

for epoch in range(1,EPOCHS+1):
    model.train(); train_losses=[]
    for images,masks in train_loader:
        images,masks=images.to(device,non_blocking=True),masks.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True); loss=loss_fn(model(images),masks); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); train_losses.append(loss.item())
    metrics=evaluate(val_loader,.2)
    row={"epoch":epoch,"train_loss":float(np.mean(train_losses)),**metrics}; history.append(row)
    print(f"{epoch:02d} train={row['train_loss']:.4f} val={row['loss']:.4f} recall={row['recall']:.3f} precision={row['precision']:.3f} dice={row['dice']:.3f}")
    if metrics["dice"]>best_dice:
        best_dice,stale=metrics["dice"],0
        torch.save({"model_state":model.state_dict(),"epoch":epoch,"validation":metrics}, OUTPUT_DIR/"best_tiny_unet_v2.pt")
    else:
        stale+=1
        if stale>=PATIENCE: print("Early stopping."); break

history_df=pd.DataFrame(history); history_df.to_csv(OUTPUT_DIR/"history.csv",index=False); display(history_df.tail())
'''),
    code(r'''
# ================================================================
# VALIDATION THRESHOLD, HELD-OUT TEST, VISUALS, ONNX, DRIVE EXPORT
# ================================================================
checkpoint=torch.load(OUTPUT_DIR/"best_tiny_unet_v2.pt",map_location=device,weights_only=False)
model.load_state_dict(checkpoint["model_state"])

thresholds=np.arange(.05,.51,.05)
threshold_results=pd.DataFrame([{"threshold":float(t),**evaluate(val_loader,float(t))} for t in thresholds])

# Balanced deployment point: chosen only using validation Dice, never test data.
chosen=threshold_results.sort_values(["dice","precision"],ascending=False).iloc[0]
chosen_threshold=float(chosen.threshold)
threshold_results.to_csv(OUTPUT_DIR/"validation_threshold_sweep.csv",index=False)
print("Balanced validation threshold:",chosen_threshold)
display(threshold_results)

test_metrics=evaluate(test_loader,chosen_threshold)
pd.DataFrame([test_metrics]).to_csv(OUTPUT_DIR/"heldout_test_metrics.csv",index=False)
print("HELD-OUT ALL-AFRICA TEST METRICS",test_metrics)

model.eval(); images,masks=next(iter(test_loader))
with torch.no_grad(): probabilities=torch.sigmoid(model(images.to(device))).cpu()
n=min(4,len(images)); fig,axes=plt.subplots(n,4,figsize=(14,3*n),squeeze=False)
for i in range(n):
    axes[i,0].imshow(images[i,0],cmap="inferno",vmin=0,vmax=1); axes[i,0].set_title("B10 normalized")
    axes[i,1].imshow(masks[i,0],cmap="gray",vmin=0,vmax=1); axes[i,1].set_title("Label")
    axes[i,2].imshow(probabilities[i,0],cmap="magma",vmin=0,vmax=1); axes[i,2].set_title("Probability")
    axes[i,3].imshow(probabilities[i,0]>=chosen_threshold,cmap="gray",vmin=0,vmax=1); axes[i,3].set_title("Prediction")
    for ax in axes[i]: ax.axis("off")
plt.tight_layout(); plt.savefig(OUTPUT_DIR/"heldout_predictions.png",dpi=160,bbox_inches="tight"); plt.show()

onnx_path=OUTPUT_DIR/"tiny_unet_africa_v2.onnx"
example=torch.zeros(1,1,256,256,device=device)
torch.onnx.export(model,example,onnx_path,input_names=["thermal_b10"],output_names=["fire_logits"],opset_version=17)

!zip -j -q /content/africa_tiny_unet_v2_results.zip "$OUTPUT_DIR"/*.pt "$OUTPUT_DIR"/*.onnx "$OUTPUT_DIR"/*.csv "$OUTPUT_DIR"/*.png
!cp /content/africa_tiny_unet_v2_results.zip /content/drive/MyDrive/
print("Saved to Drive: /content/drive/MyDrive/africa_tiny_unet_v2_results.zip")
'''),
]

output = Path(__file__).resolve().parents[1] / "notebooks" / "03_train_africa_balanced_tiny_unet.ipynb"
nbf.write(nb, output)
print(output)

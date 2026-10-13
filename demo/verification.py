import sys; sys.path.insert(0, ".")
import app
for name, f in app.PRESET_SAMPLES.items():
    gt = app.find_ground_truth_mask(f)
    print(f"{name[:45]:45s}", None if gt is None else int(gt.sum()))
"""Write the result CSV tables from results/metrics.json (no recomputation)."""
import csv
import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
R = json.loads((RES / "metrics.json").read_text())
for name, rows in (("e1_models", R["E1_models_ascending"]), ("e4_sensitivity", R["E4_sensitivity"]),
                   ("e5_landcover", R["E5_landcover_ascending"])):
    with open(RES / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
s = R["E1_E2_summary"]
for k, v in s.items():
    print(k, {m: round(v[m], 3) for m in ("pearson_r", "ssim", "shift_lines", "shift_pixels", "rmse_db_bias_removed")})
print("geocoding", R["E3_geocoding_vs_esa"])
print("classes", json.dumps(R["E3_real_sigma0_by_predicted_class"], indent=0)[:1500])

"""Merge full_n_cv + full_n_combined_test from the preserved backup
into the freshly-regenerated cf_targeted_7km.json."""
import json
from pathlib import Path

NEW = Path("/opt/induced-seismicity/cf_targeted_7km.json")
PRIOR = Path("/opt/induced-seismicity/cf_targeted_7km.json.preserve_full_n")

new = json.loads(NEW.read_text())
prior = json.loads(PRIOR.read_text())
print(f"new keys: {list(new.keys())}")
print(f"prior keys: {list(prior.keys())}")

for k in ("full_n_cv", "full_n_combined_test"):
    if k in prior:
        new[k] = prior[k]
        print(f"  merged: {k}")

NEW.write_text(json.dumps(new, indent=2))
print(f"final keys: {list(new.keys())}")

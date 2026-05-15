import subprocess
import sys
import os
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

scripts = [
    r"C:\Project Work\Scripts_nobpfo100\generate_splits_nobpfo100.py",
    r"C:\Project Work\Scripts_nobpfo100\precompute_stft_nobpfo100.py",
    r"C:\Project Work\Scripts_nobpfo100\precompute_dwt_nobpfo100.py",
    r"C:\Project Work\Scripts_nobpfo100\precompute_envelope_nobpfo100.py",
    r"C:\Project Work\Scripts_nobpfo100\train_stft_cnn_nobpfo100.py",
    r"C:\Project Work\Scripts_nobpfo100\train_dwt_cnn_nobpfo100.py",
    r"C:\Project Work\Scripts_nobpfo100\train_envelope_cnn_reg_nobpfo100.py",
    r"C:\Project Work\Scripts_nobpfo100\train_envelope_cnn_dilated_nobpfo100.py",
    r"C:\Project Work\Scripts_nobpfo100\compute_logits_nobpfo100.py",
    r"C:\Project Work\Scripts_nobpfo100\ensemble_evaluate_nobpfo100.py",
    r"C:\Project Work\Scripts_nobpfo100\ensemble_evaluate_temperature_nobpfo100.py",
]

out_dir = Path(r"C:\Project Work\Outputs_nobpfo100")
out_dir.mkdir(parents=True, exist_ok=True)
log_file = out_dir / "run.log"

print(f"--- STARTING nobpfo100 PIPELINE ---", flush=True)
with open(log_file, "w", encoding="utf-8") as logf:
    for script in scripts:
        name = Path(script).name
        print(f"  -> {name}", flush=True)
        logf.write(f"\n{'='*70}\n>>> RUNNING {name}\n{'='*70}\n")
        logf.flush()

        res = subprocess.run([sys.executable, script],
                             capture_output=True, text=True, encoding="utf-8")

        logf.write(res.stdout)
        logf.write(res.stderr)
        logf.flush()

        if res.returncode != 0:
            print(f"  FAILED {name} (see {log_file})", flush=True)
            sys.exit(1)

print("ALL DONE", flush=True)

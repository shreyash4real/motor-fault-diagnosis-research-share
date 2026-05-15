"""
Helper script to extract logits from the trained ENVELOPE_resnet_v2 model.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

sys.path.insert(0, r'C:\Project Work\Scripts_nobpfo100')
from train_envelope_cnn_v2_nobpfo100 import load_pt_dataset, EnvelopeResNet1D, BATCH_SIZE, USE_AMP

def infer_and_save(split, model, loader, ds, device, out_dir):
    model.eval()
    all_logits, all_labels, all_indices = [], [], []
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device)
            with autocast('cuda', enabled=USE_AMP):
                logits = model(x)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.numpy())
            all_indices.append(idx.numpy())
            
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    indices = np.concatenate(all_indices)
    
    np.save(out_dir / f"{split}_logits.npy", logits.astype(np.float32))
    np.save(out_dir / f"{split}_labels.npy", labels.astype(np.int64))
    
    rows = []
    for i in indices:
        md = ds.metadata[i]
        rows.append({
            "ch1_path": md["ch1_path"],
            "col_index": md["col_index"],
            "seg_idx": md["seg_idx"]
        })
    pd.DataFrame(rows).to_csv(out_dir / f"{split}_meta_sigs.csv", index=False)
    print(f"Saved {split} logits: {logits.shape}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(r'C:\Project Work\Outputs_nobpfo100\training\ENVELOPE_resnet_v2')
    features_dir = Path(r'C:\Project Work\Outputs_nobpfo100\features\envelope')

    val_ds = load_pt_dataset('val', features_dir)
    test_ds = load_pt_dataset('test', features_dir)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = EnvelopeResNet1D(num_classes=4).to(device)
    ckpt = torch.load(out_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(ckpt['model_state'])

    infer_and_save("val", model, val_loader, val_ds, device, out_dir)
    infer_and_save("test", model, test_loader, test_ds, device, out_dir)

if __name__ == "__main__":
    main()

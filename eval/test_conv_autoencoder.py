import importlib
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from spectral_frontend.datasets.dataset import QuantizedLogAgcDataset
from spectral_frontend.datasets.window_dataset import WindowedLogAgcDataset
from torch.utils.data import DataLoader


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


import re


def export_voxfe_weights(model, path="voxfe/bulk/voxfe_weights.npz"):
    weights = {}
    for name, param in model.named_parameters():
        if name.startswith("encoder."):
            weights[name] = param.detach().cpu().numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **weights)
    print(f"✅ Saved Voxfe weights to: {path}")


def main():
    cfg = load_config("spectral_frontend/configs/basic_config.yaml")
    tr_cfg = cfg["training"]
    pcen_cfg = cfg["signal_processing"]["pcen"]
    base_dir = cfg["preprocessing"]["dataset_base_dir"]
    data_dir = os.path.join(base_dir, tr_cfg["dataset_name"])
    ckpt_dir = Path(tr_cfg["checkpoint_dir"])

    # Dataset
    val_set = QuantizedLogAgcDataset(
        data_dir,
        bitdepth=pcen_cfg["bitdepth"],
        min_db=pcen_cfg["min_db"],
        max_db=pcen_cfg["max_db"],
        split="val",
        device="cuda",
    )
    window_size = tr_cfg["train_window_frames"]
    val_ds = WindowedLogAgcDataset(val_set.data, window_size)
    val_loader = DataLoader(val_ds, batch_size=tr_cfg["batch_size"], shuffle=False)

    # Model loading
    model_name = tr_cfg["model_name"]
    model_module = tr_cfg["model_module"]
    model_mod = importlib.import_module(f"spectral_frontend.{model_module}")
    model_cls = getattr(model_mod, model_name)
    model = model_cls(latent_dim=tr_cfg["latent_dim"]).cuda()

    def extract_epoch_num(path):
        match = re.search(r'epoch(\d+)\.pt$', path.name)
        return int(match.group(1)) if match else -1

    ckpt_files = sorted(ckpt_dir.glob("epoch*.pt"), key=extract_epoch_num)
    # Load last checkpoint
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
    last_ckpt = ckpt_files[-1]
    state = torch.load(last_ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    print(f"✅ Loaded model from {last_ckpt}")

    export_voxfe_weights(model, path="voxfe/bulk/voxfe_weights.npz")

    print(f"✅ Saved model as reference for the cupy encder")

    # Output buffers
    all_inputs = []
    all_latents = []

    with torch.no_grad():
        for i, (x, _) in enumerate(val_loader):
            x = x.cuda()
            _, latents = model(x)  # skip recon, just get latents

            all_inputs.append(x.cpu().numpy())
            all_latents.append(latents.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"→ Processed batch {i + 1}/{len(val_loader)}")

    # Save result
    out_dir = ckpt_dir / "test_output"
    out_dir.mkdir(exist_ok=True)

    inputs = np.concatenate(all_inputs, axis=0)
    latents = np.concatenate(all_latents, axis=0)

    np.savez(out_dir / "val_inputs_and_latents.npz", inputs=inputs, latents=latents)
    print(f"✅ Saved inputs + latents to {out_dir / 'val_inputs_and_latents.npz'}")


if __name__ == "__main__":
    main()

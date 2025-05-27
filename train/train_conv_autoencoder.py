# blurt/train_conv_autoencoder.py

import inspect
import os
import shutil
import time
from pathlib import Path

import torch
import yaml
from spectral_frontend.datasets.dataset import QuantizedLogAgcDataset
from spectral_frontend.datasets.window_dataset import WindowedLogAgcDataset
from spectral_frontend.models import get_model_class
from spectral_frontend.recording.training_recorder import TrainingRecorder
from torch.utils.data import DataLoader


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("spectral_frontend/configs/basic_config.yaml")
    tr_cfg = cfg["training"]
    pcen_cfg = cfg["signal_processing"]["pcen"]
    base_dir = cfg["preprocessing"]["dataset_base_dir"]
    data_dir = os.path.join(base_dir, cfg["training"]["dataset_name"])

    # Load dataset into GPU
    train_set = QuantizedLogAgcDataset(
        data_dir,
        bitdepth=pcen_cfg["bitdepth"],
        min_db=pcen_cfg["min_db"],
        max_db=pcen_cfg["max_db"],
        split="train",
        device="cuda"
    )

    val_set = QuantizedLogAgcDataset(
        data_dir,
        bitdepth=pcen_cfg["bitdepth"],
        min_db=pcen_cfg["min_db"],
        max_db=pcen_cfg["max_db"],
        split="val",
        device="cuda"
    )

    # Wrap in windowed loader
    window_size = tr_cfg["train_window_frames"]
    train_ds = WindowedLogAgcDataset(train_set.data, window_size)
    val_ds = WindowedLogAgcDataset(val_set.data, window_size)
    train_loader = DataLoader(train_ds, batch_size=tr_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tr_cfg["batch_size"], shuffle=False)

    # --- Model loading ---
    model_key = tr_cfg["model_name"]  # e.g. "4L12DW"
    model_cls = get_model_class(model_key)
    model = model_cls(latent_dim=tr_cfg["latent_dim"]).cuda()

    # --- Optimizer and scheduler ---
    lr = float(tr_cfg["learning_rate"])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = tr_cfg["num_epochs"]
    save_every = tr_cfg.get("save_every", 5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    # --- Recorder and setup ---
    ckpt_dir = Path(tr_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Copy YAML config
    shutil.copyfile("spectral_frontend/configs/basic_config.yaml", ckpt_dir / "config.yaml")
    print(f"📄 Copied config to {ckpt_dir / 'config.yaml'}")
    # Copy model source file
    model_src = Path(inspect.getfile(model_cls))
    shutil.copyfile(model_src, ckpt_dir / model_src.name)
    print(f"📄 Copied model code to {ckpt_dir / model_src.name}")

    recorder = TrainingRecorder(checkpoint_dir=tr_cfg["checkpoint_dir"], config_dict=cfg)
    recorder_batch_interval = 2000
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()

        # save additional data during training
        y_buffer = []
        recon_buffer = []
        train_loss_buffer = []
        val_loss_buffer = []
        latents_buffer = []

        train_loss = 0.0

        for i, (x, y) in enumerate(train_loader, 1):
            x, y = x.cuda(), y.cuda()  # x: [B, 1, 4, 180], y: [B, 180]
            recon, latents = model(x)
            loss = model.loss(recon, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()

            if i % recorder_batch_interval == 0:
                # save y (target) and recon (decoder output)
                y_buffer.append(y.detach().cpu().numpy())
                recon_buffer.append(recon.detach().cpu().numpy())
                train_loss_buffer.append(loss.item())
                latents_buffer.append(latents.detach().cpu().numpy())  # [B, latent_dim]

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for j, (x, y) in enumerate(val_loader, 1):
                x, y = x.cuda(), y.cuda()
                recon, _ = model(x)
                loss = model.loss(recon, y)
                val_loss += loss.item()

                if j % recorder_batch_interval == 0:
                    val_loss_buffer.append(loss.item())

        val_loss /= len(val_loader)

        scheduler.step()  # ✅ Step the scheduler here
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - start_time
        elapsed_str = f"{elapsed:.0f}s" if elapsed < 3600 else f"{elapsed / 3600:.2f}h"
        print(f"[Epoch {epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.6e} | "
              f"Time: {elapsed_str}")

        if epoch % save_every == 0:
            recorder.save_epoch(epoch, model, opt, scheduler, y_buffer, recon_buffer,
                                latents_buffer=latents_buffer,
                                train_loss_buffer=train_loss_buffer,
                                val_loss_buffer=val_loss_buffer)

            y_buffer.clear()
            recon_buffer.clear()
            latents_buffer.clear()  # ✅ ADD THIS LINE
            train_loss_buffer.clear()
            val_loss_buffer.clear()


if __name__ == "__main__":
    main()

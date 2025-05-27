"""
train_autoencoder_ddp.py
-------------------------

Trains the SpectralConvAutoencoder model using pre-extracted PCEN windows,
stored as .npy shard files (e.g. shape [N, 6, 180], dtype=uint16), using
a ping-pong preloading buffer for improved throughput.

Supports both single-GPU and multi-GPU (DDP) execution.

Usage:
  Single GPU:
    python train_autoencoder_ddp.py

  Multi-GPU (e.g., 4 GPUs):
    torchrun --nproc_per_node=4 train_autoencoder_ddp.py
"""

import inspect
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from spectral_frontend.models import get_model_class
from spectral_frontend.recording.training_recorder import TrainingRecorder
from spectral_frontend.utils.custom_dataset import CustomDatasetFromArray
from spectral_frontend.utils.pingpong_loader import ShardPingPongLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo"
        )
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    return rank, local_rank, world_size


def main():
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    cfg = load_config("spectral_frontend/configs/basic_config.yaml")
    tr_cfg = cfg["training"]

    pcen_norm = cfg["signal_processing"]["pcen"].get("pcen_normalization", {})
    if pcen_norm.get("method") != "fixed":
        raise ValueError("Expected 'method: fixed' under pcen_normalization.")
    pcen_mean = float(pcen_norm["mean"])
    pcen_std = float(pcen_norm["std"])

    # --- Directories ---
    base_dir = Path(cfg["preprocessing"]["dataset_base_dir"])
    shards_dir = base_dir / cfg["preprocessing"]["shards_dir"]
    train_dir = shards_dir / cfg["preprocessing"]["shard_split_output_dirs"]["train"]
    val_dir = shards_dir / cfg["preprocessing"]["shard_split_output_dirs"]["val"]
    val_files = sorted(val_dir.glob("shard_*.npy"))

    # --- Shard shape (read from file header) ---
    with val_files[0].open('rb') as f:
        version = np.lib.format.read_magic(f)
        shard_shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)

    # --- Ping-pong buffer setup ---
    shards_per_proc = cfg["preprocessing"]["shards_per_proc"]
    nproc_per_node = cfg["preprocessing"]["nproc_per_node"]
    shards_per_node = shards_per_proc * nproc_per_node
    train_shard_files = sorted(train_dir.glob("shard_*.npy"))

    shard_loader = ShardPingPongLoader(
        shard_paths=train_shard_files,
        shards_per_proc=shards_per_proc,
        shard_shape=shard_shape
    )
    shard_loader.start()

    # --- Validation loader (fixed single shard) ---
    val_dataset = CustomDatasetFromArray(np.load(val_files[0]), pcen_mean, pcen_std)
    val_loader = DataLoader(
        val_dataset,
        batch_size=tr_cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    # --- Model ---
    model_key = tr_cfg["model_name"]
    model_cls = get_model_class(model_key)
    model = model_cls(latent_dim=tr_cfg["latent_dim"]).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # --- Optimizer / Scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tr_cfg["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tr_cfg["num_epochs"])

    # --- Checkpointing ---
    if rank == 0:
        ckpt_dir = Path(tr_cfg["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile("spectral_frontend/configs/basic_config.yaml", ckpt_dir / "config.yaml")
        model_src = Path(inspect.getfile(model_cls))
        shutil.copyfile(model_src, ckpt_dir / model_src.name)
        print(f"📄 Copied model + config to {ckpt_dir}")
        recorder = TrainingRecorder(checkpoint_dir=ckpt_dir, config_dict=cfg)
    else:
        recorder = None

    # --- Training Loop ---
    start = time.time()
    nshards = len(train_shard_files)

    for epoch in range(1, tr_cfg["num_epochs"] + 1):
        shard_idx = 0
        for _ in range(0, nshards, shards_per_proc):
            shard_batch = shard_loader.get_next_buffer()  # shape: (shards_per_proc, N, 6, 180)

            for shard in shard_batch:
                train_dataset = CustomDatasetFromArray(shard, pcen_mean, pcen_std)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=tr_cfg["batch_size"],
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True,
                    drop_last=False
                )

                model.train()
                shard_loss = 0.0

                for x, y in train_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    recon, latents = model(x)
                    loss = model.loss(recon, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    shard_loss += loss.item()

                shard_loss /= len(train_loader)

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x, y in val_loader:
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        recon, _ = model(x)
                        loss = model.loss(recon, y)
                        val_loss += loss.item()
                val_loss /= len(val_loader)

                if rank == 0:
                    print(f"[Epoch {epoch}/{tr_cfg['num_epochs']}] "
                          f"Shard {shard_idx}/{nshards} | "
                          f"Train Loss: {shard_loss:.6f} | "
                          f"Val Loss: {val_loss:.6f} | "
                          f"LR: {scheduler.get_last_lr()[0]:.2e}")
                shard_idx += 1

        scheduler.step()

    shard_loader.stop()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

# """
# train_autoencoder_ddp.py
# -------------------------
#
# Trains the SpectralConvAutoencoder model using pre-extracted PCEN windows,
# stored as .npy shard files (e.g. shape [N, 6, 180], dtype=uint16).
#
# Supports both single-GPU and multi-GPU (DDP) execution.
#
# Usage:
#   Single GPU:
#     python train_autoencoder_ddp.py
#
#   Multi-GPU (2 GPUs):
#     torchrun --nproc_per_node=2 train_autoencoder_ddp.py
# """
#
# import inspect
# import os
# import shutil
# import time
# from pathlib import Path
#
# import numpy as np
# import torch
# import torch.distributed as dist
# import yaml
# from spectral_frontend.models import get_model_class
# from spectral_frontend.recording.training_recorder import TrainingRecorder
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import Dataset, DataLoader
#
#
# # --------------
# # Dataset Loader
# # --------------
# class SingleShardDataset(Dataset):
#     def __init__(self, shard_path, mean, std):
#         self.data = np.load(shard_path, mmap_mode="r")
#         self.length = self.data.shape[0]
#         self.mean = mean
#         self.std = std
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, idx):
#         x = self.data[idx].astype(np.float32) / 4095.0  # → [0, 1]
#         x = (x - self.mean) / self.std
#         y = (x[-1])  # target: normalized last frame
#         x = x[np.newaxis, ...]  # shape: [1, 6, 180]
#         return torch.from_numpy(x), torch.from_numpy(y)
#
#
# def init_ddp():
#     dist.init_process_group(backend='nccl')
#     local_rank = int(os.environ["LOCAL_RANK"])
#     torch.cuda.set_device(local_rank)
#     return dist.get_rank(), local_rank, dist.get_world_size()
#
#
# # ------------------
# # Training Utilities
# # ------------------
# def load_config(config_path):
#     with open(config_path, "r") as f:
#         return yaml.safe_load(f)
#
#
# def setup_ddp():
#     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         dist.init_process_group(
#             backend="nccl" if torch.cuda.is_available() else "gloo"
#         )
#         rank = dist.get_rank()
#         local_rank = int(os.environ["LOCAL_RANK"])
#         world_size = dist.get_world_size()
#     else:
#         rank = 0
#         local_rank = 0
#         world_size = 1
#     return rank, local_rank, world_size
#
#
# # -----
# # Main
# # -----
# def main():
#     rank, local_rank, world_size = setup_ddp()
#     device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
#
#     cfg = load_config("spectral_frontend/configs/basic_config.yaml")
#     tr_cfg = cfg["training"]
#
#     pcen_norm = cfg["signal_processing"]["pcen"].get("pcen_normalization", {})
#     if pcen_norm.get("method") != "fixed":
#         raise ValueError("Expected 'method: fixed' under pcen_normalization.")
#     pcen_mean = float(pcen_norm["mean"])
#     pcen_std = float(pcen_norm["std"])
#
#     # --- Load datasets ---
#     base_dir = Path(cfg["preprocessing"]["dataset_base_dir"])
#     shards_dir = base_dir / cfg["preprocessing"]["shards_dir"]
#     train_dir = shards_dir / "train"
#     val_dir = shards_dir / "val"
#     val_files = sorted(val_dir.glob("shard_*.npy"))
#
#     # extract shape of a shard
#     shards_per_proc = cfg["preprocessing"]["shards_per_pro"]
#     nproc_per_node = cfg["preprocessing"]["nproc_per_node"]
#     shards_per_node = shards_per_proc * nproc_per_node
#     with val_files[0].open('rb') as f:
#         version = np.lib.format.read_magic(f)
#         shard_shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
#         nwindows, nframe, nelem = shard_shape
#
#     # allocate the pingpong CPU buffer
#     ping_pong_shape = ((2, shards_per_node) + shard_shape)
#     ping_pong_buffer = np.empty(ping_pong_shape, dtype=np.uint16)
#
#     # val_set will be fixed, use first shard (or extend later to loop)
#     val_dataset = SingleShardDataset(val_files[0], pcen_mean, pcen_std)  # load 1 shard for validation
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=tr_cfg["batch_size"],
#         shuffle=False,
#         num_workers=2,
#         pin_memory=True,
#         drop_last=False
#     )
#
#     # prepare list of training shard paths for per-epoch use
#     train_dir = shards_dir / "train"
#     train_shard_files = sorted(train_dir.glob("pcen_shard_*.npy"))
#
#     # manually extract mean,std from
#     # arr = np.load(train_shard_files[0]).astype(np.float32) / 4095.  # shape: [N, 6, 180]
#
#     # --- Model ---
#     model_key = tr_cfg["model_name"]
#     model_cls = get_model_class(model_key)
#     model = model_cls(latent_dim=tr_cfg["latent_dim"]).to(device)
#
#     if world_size > 1:
#         model = DDP(model, device_ids=[local_rank])
#
#     # --- Optimizer / Scheduler ---
#     optimizer = torch.optim.Adam(model.parameters(), lr=float(tr_cfg["learning_rate"]))
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tr_cfg["num_epochs"])
#
#     # --- Checkpoint setup ---
#     if rank == 0:
#         ckpt_dir = Path(tr_cfg["checkpoint_dir"])
#         ckpt_dir.mkdir(parents=True, exist_ok=True)
#         shutil.copyfile("spectral_frontend/configs/basic_config.yaml", ckpt_dir / "config.yaml")
#         model_src = Path(inspect.getfile(model_cls))
#         shutil.copyfile(model_src, ckpt_dir / model_src.name)
#         print(f"📄 Copied model + config to {ckpt_dir}")
#
#         recorder = TrainingRecorder(checkpoint_dir=ckpt_dir, config_dict=cfg)
#     else:
#         recorder = None  # other ranks don't save
#
#     # --- Training Loop ---
#     # torch.cuda.synchronize()
#     start = time.time()
#
#     for epoch in range(1, tr_cfg["num_epochs"] + 1):
#         for shard_idx, shard_path in enumerate(train_shard_files):
#             train_dataset = SingleShardDataset(shard_path, pcen_mean, pcen_std)
#             train_loader = DataLoader(
#                 train_dataset,
#                 batch_size=tr_cfg["batch_size"],
#                 shuffle=True,
#                 num_workers=2,
#                 pin_memory=True,
#                 drop_last=False
#             )
#
#             model.train()
#             shard_loss = 0.0
#
#             for x, y in train_loader:
#                 x, y = x.to(device), y.to(device)
#                 recon, latents = model(x)
#                 loss = model.loss(recon, y)
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#                 shard_loss += loss.item()
#
#             shard_loss /= len(train_loader)
#
#             # --- Validation after each shard ---
#             model.eval()
#             val_loss = 0.0
#             with torch.no_grad():
#                 for x, y in val_loader:
#                     x, y = x.to(device), y.to(device)
#                     recon, _ = model(x)
#                     loss = model.loss(recon, y)
#                     val_loss += loss.item()
#
#             val_loss /= len(val_loader)
#
#             if rank == 0:
#                 print(f"[Epoch {epoch}/{tr_cfg['num_epochs']}] "
#                       f"Shard {shard_idx}/{len(train_shard_files)} | "
#                       f"Train Loss: {shard_loss:.6f} | "
#                       f"Val Loss: {val_loss:.6f} | "
#                       f"LR: {scheduler.get_last_lr()[0]:.2e}")
#
#                 # if epoch % tr_cfg.get("save_every", 5) == 0:
#                 #     recorder.record_shard(
#                 #         y=y.detach().cpu().numpy(),
#                 #         recon=recon.detach().cpu().numpy(),
#                 #         latents=latents.detach().cpu().numpy(),
#                 #         train_loss=loss.item(),
#                 #         val_loss=val_loss
#                 #     )
#
#             print(f"Shard train time: {time.time() - start:.2f} s")
#         # recorder.save_epoch(epoch, model, optimizer, scheduler)
#         scheduler.step()  # step the scheduler per epoch
#         # torch.cuda.synchronize()
#
#     # --- Cleanup ---
#     if world_size > 1:
#         dist.destroy_process_group()
#
#
# if __name__ == "__main__":
#     main()

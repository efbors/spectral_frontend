import datetime
import glob
import importlib.util
import os
import re
from pathlib import Path

import imageio
import matplotlib
import torch.nn as nn
import yaml
from PIL import Image

matplotlib.use("Agg")  # Use non-GUI backend to avoid flashes
import matplotlib.pyplot as plt


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_checkpoint_paths(base_dir):
    path_pattern = os.path.join(base_dir, "epoch*.pt")
    paths = glob.glob(path_pattern)

    # Extract epoch numbers and sort numerically
    def extract_epoch_num(path):
        match = re.search(r'epoch(\d+)\.pt$', os.path.basename(path))
        return int(match.group(1)) if match else -1

    paths = sorted(paths, key=extract_epoch_num)

    print(f"Found {len(paths)} checkpoint(s) in numerical order.")
    return paths


def load_model_class_from_checkpoint_dir(ckpt_dir):
    model_py_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".py")]
    assert len(model_py_files) == 1, f"Expected one model .py file, found: {model_py_files}"

    model_path = os.path.join(ckpt_dir, model_py_files[0])
    module_name = os.path.splitext(model_py_files[0])[0]

    spec = importlib.util.spec_from_file_location(module_name, model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # Find the first class in the module (assumes only one model class defined)
    for attr in dir(model_module):
        obj = getattr(model_module, attr)
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            return obj

    raise RuntimeError(f"No nn.Module class found in {model_path}")


def load_latents_from_paths(paths):
    """
    Load latents from checkpoint paths, keeping each epoch separate.
    Returns: ndarray of shape (nepochs, nbatch, latent_dim)
    """
    latents = []
    for path in paths:
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            if "latents" in state:
                latent_epoch = np.array(state["latents"])  # shape: (batch, latent_dim)
                latents.append(latent_epoch)
            else:
                print(f"⚠️ No 'latents' in {path}")
        except Exception as e:
            print(f"⚠️ Skipping {path}: {e}")

    if latents:
        return np.stack(latents, axis=0)  # shape: (nepochs, batch, latent_dim)
    else:
        return np.zeros((0, 1, 1))


def load_recon_from_paths(paths):
    """
    Load reconstructions from checkpoint paths, keeping each epoch separate.
    Returns: ndarray of shape (nepochs, nbatch, last_dim)
    """
    recon = []
    for path in paths:
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            if "recon" in state:
                recon_epoch = np.array(state["recon"])  # shape: (batch, dim)
                recon.append(recon_epoch)
            else:
                print(f"⚠️ No 'recon' in {path}")
        except Exception as e:
            print(f"⚠️ Skipping {path}: {e}")

    if recon:
        return np.stack(recon, axis=0)  # shape: (nepochs, batch, dim)
    else:
        return np.zeros((0, 1, 1))


def load_y_from_paths(paths):
    """
    Load target outputs ('y') from checkpoint paths, preserving epoch structure.
    Returns: ndarray of shape (nepochs, nbatch, last_dim)
    """
    y_list = []
    for path in paths:
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            if "y" in state:
                y_epoch = np.array(state["y"])  # shape: (batch, dim)
                y_list.append(y_epoch)
            else:
                print(f"⚠️ No 'y' in {path}")
        except Exception as e:
            print(f"⚠️ Skipping {path}: {e}")

    if y_list:
        return np.stack(y_list, axis=0)  # shape: (nepochs, batch, dim)
    else:
        return np.zeros((0, 1, 1))


from sklearn.decomposition import PCA


def get_global_pca_limits_from_latents(latents, max_samples=10000):
    """
    Compute global axis limits for PCA projections (pc1_2, pc2_3, pc3_4)
    over all epochs of latent vectors.

    Args:
        latents: ndarray, shape (nepoch, nbatch, latent_dim)
        max_samples: int, limit number of samples used in PCA for performance

    Returns:
        limits: dict with keys "pc1_2", "pc2_3", "pc3_4", each containing:
                {"xlim": [min, max], "ylim": [min, max]}
    """
    all_proj = {"pc1_2": [], "pc2_3": [], "pc3_4": []}
    nepoch, nbatch, latent_dim = latents.shape

    # Flatten across all epochs and batches
    latents_flat = latents.reshape(-1, latent_dim)

    # Subsample if needed
    if latents_flat.shape[0] > max_samples:
        indices = np.random.choice(latents_flat.shape[0], max_samples, replace=False)
        latents_flat = latents_flat[indices]

    # Center the data
    latents_flat -= latents_flat.mean(axis=0)

    # Run PCA
    pca = PCA()
    X_pca = pca.fit_transform(latents_flat)

    # Collect projections
    all_proj["pc1_2"].append(X_pca[:, [0, 1]])
    all_proj["pc2_3"].append(X_pca[:, [1, 2]])
    all_proj["pc3_4"].append(X_pca[:, [2, 3]])

    # Compute axis limits for each PCA projection
    limits = {}
    for key, proj_list in all_proj.items():
        X = np.vstack(proj_list)
        limits[key] = {
            "xlim": [X[:, 0].min(), X[:, 0].max()],
            "ylim": [X[:, 1].min(), X[:, 1].max()],
        }

    return limits


def make_gif_from_pca_series(prefix_path=None, pattern=None, output_path=None, fps=1):
    """
    Generate a GIF from a series of PCA plot PNGs.
    - prefix_path + pattern determines file matching and default .gif output name
    - If output_path is not given, it's derived from prefix + cleaned pattern

    Example:
        prefix_path=".../latent_PCA_250515_250514a_vctk"
        pattern="_scatter_combined_*.png"
        → output: latent_PCA_250515_250514a_vctk_scatter_combined.gif
    """
    if prefix_path and pattern:
        search_pattern = os.path.join(prefix_path, pattern)
        if output_path is None:
            base = Path(prefix_path).with_suffix('')
            pattern_tag = pattern.replace("*", "").replace(".png", "").strip("_")
            output_gif = str(base) + f"_{pattern_tag}.gif"
        else:
            output_gif = output_path
    elif pattern:
        search_pattern = pattern
        output_gif = output_path or "pca_animation.gif"
    else:
        raise ValueError("You must provide either (prefix_path + pattern) or a full glob pattern.")

    paths = sorted(glob.glob(search_pattern))
    if not paths:
        print(f"⚠️ No matching frames found for pattern: {search_pattern}")
        return

    base_size = Image.open(paths[0]).size
    frames = []
    for path in paths:
        img = Image.open(path)
        if img.size != base_size:
            img = img.resize(base_size, Image.BICUBIC)
        frames.append(np.array(img))

    imageio.mimsave(output_gif, frames, fps=fps)
    print(f"✅ Saved animated GIF to: {output_gif}")


def extract_conv_weights(state, module_name, layer_idx):
    model_state = state["model_state_dict"]
    weights = None
    for key in model_state:
        if f"{module_name}." in key and str(layer_idx) in key and ".weight" in key:
            weights = model_state[key]
            break
    if weights is None:
        raise KeyError(f"No conv{layer_idx} weight found in model_state_dict")
    return weights  # shape: [out_ch, in_ch, 3, 3]


def compute_cosine_sims(weights_list):
    sims = []
    for t in range(len(weights_list) - 1):
        w1 = weights_list[t]  # [out_ch, in_ch, kH, kW]
        w2 = weights_list[t + 1]

        # Validate shapes match
        assert w1.shape == w2.shape, f"Shape mismatch: {w1.shape} vs {w2.shape}"

        # Flatten the 3x3 kernels for cosine similarity
        w1_flat = w1.view(w1.shape[0], w1.shape[1], -1)  # [out_ch, in_ch, k*k]
        w2_flat = w2.view(w2.shape[0], w2.shape[1], -1)

        cos = torch.nn.functional.cosine_similarity(w1_flat, w2_flat, dim=2)  # [out_ch, in_ch]
        sims.append(cos)
    return sims  # List of [out_ch, in_ch]


def plot_epoch_cosines(cosine_sims, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, sim in enumerate(cosine_sims):
        plt.figure(figsize=(10, 8))
        plt.imshow(sim.numpy(), vmin=0, vmax=1, cmap='plasma', aspect='auto')
        plt.title(f"Cosine Similarity: Epoch {i} → {i + 1}")
        plt.xlabel("Input Channel")
        plt.ylabel("Output Channel")
        plt.colorbar(label="Cosine Similarity")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"cosine_epoch_{i:02d}.png"))
        plt.close()


def plot_cosine_summary(cosine_sims, save_path="cosine_summary.png"):
    num_plots = len(cosine_sims)  # 63 for epochs 0→1 through 62→63
    rows, cols = 9, 7  # adjust if you prefer a different layout
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), dpi=150)

    for idx, sim in enumerate(cosine_sims):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        im = ax.imshow(sim.numpy(), vmin=0, vmax=1, cmap='plasma', aspect='auto')
        ax.set_title(f"{idx}→{idx + 1}", fontsize=8)
        ax.axis('off')

    # Remove empty plots (if any)
    for i in range(num_plots, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].axis('off')

    # Add one shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.8])
    fig.colorbar(im, cax=cbar_ax, label="Cosine Similarity")

    plt.suptitle("Cosine Similarity per Epoch Transition", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved summary image to {save_path}")


def plot_latent_variance(ckpt_paths, save_path="latent_variance.png"):
    latent_list = []
    for path in ckpt_paths:
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            latents = state.get("latents")
            if latents is not None:
                latent_list.append(latents)  # [N, latent_dim]
        except Exception as e:
            print(f"⚠️ Skipping {path}: {e}")

    if not latent_list:
        print("❌ No latent data found.")
        return

    all_latents = np.concatenate(latent_list, axis=0)
    epochs = len(latent_list)

    # Compute variance per dimension per epoch
    latent_vars = np.array([np.var(l, axis=0) for l in latent_list])  # [epochs, latent_dim]

    plt.figure(figsize=(12, 6))
    plt.imshow(latent_vars.T, aspect='auto', cmap='cividis', interpolation='nearest')
    # plt.imshow(latent_vars.T, aspect='auto', cmap='magma', interpolation='nearest')
    plt.colorbar(label='Variance')
    plt.xlabel("Epoch")
    plt.ylabel("Latent Dimension")
    plt.title("Latent Variance Over Epochs")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved latent variance plot to: {save_path}")


import numpy as np
import torch


def compute_latent_entropy(latents: np.ndarray, bins=64) -> dict:
    """
    Compute per-dimension and total entropy from a 2D latent array.
    (this is also a proxy for I(Z;X) in IB)
    latents: [N, latent_dim]
    Returns: dict with keys:
        'entropy_per_dim': [latent_dim]
        'total_entropy': scalar

    """
    N, D = latents.shape
    entropy = np.zeros(D)

    for i in range(D):
        hist, _ = np.histogram(latents[:, i], bins=bins, density=True)
        p = hist + 1e-12
        p /= p.sum()
        entropy[i] = -np.sum(p * np.log2(p))

    return {
        'entropy_per_dim': entropy,
        'total_entropy': float(np.sum(entropy)),
    }


def plot_latent_entropy_bars(entropy_dict, save_path="latent_entropy.png"):
    entropy = entropy_dict['entropy_per_dim']
    total = entropy_dict['total_entropy']
    num_dims = len(entropy)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[4, 1], sharex=True)

    axes[0].bar(np.arange(num_dims), entropy)
    axes[0].set_ylabel("Entropy (bits)")
    axes[0].set_title("Latent Entropy per Dimension")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].bar([0], [total], width=0.4)
    axes[1].set_ylabel("Total")
    axes[1].set_ylim(0, num_dims * np.log2(64))  # max range for 64 bins
    axes[1].set_xticks([])
    axes[1].set_title(f"Cumulative Latent Entropy: {total:.2f} bits")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved latent entropy plot to: {save_path}")


def plot_latent_pca_eigenvalues_epoch(latents, prefix_path, suffix, max_components=20):
    latents = latents - latents.mean(axis=0)
    pca = PCA()
    pca.fit(latents)

    eigvals = pca.explained_variance_ratio_[:max_components] * 100  # in %
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(eigvals) + 1), eigvals, marker="o")

    ax.set_title(f"PCA Explained Variance — Epoch {suffix}")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_ylim(0, max(5, eigvals[0] * 1.2))  # adaptive height if low-variance
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_path = os.path.join(prefix_path, f"eigen_{suffix}.png")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"✅ Saved PCA eigenvalue plot: {out_path}")


def plot_latent_pca_epoch_scatter(latents_epoch, prefix_path, suffix, pca_limits, max_samples=10000):
    """
    Generate a composite PCA scatter plot for one epoch using PCs 1–4.

    Args:
        latents_epoch: ndarray [nbatch, latent_dim]
        prefix_path: full directory where PNGs will be stored
        suffix: string identifier (e.g., "03")
        pca_limits: dict of {"pc1_2": {xlim, ylim}, ...}
    """
    os.makedirs(prefix_path, exist_ok=True)  # Ensure output directory exists

    if latents_epoch.shape[0] == 0:
        return

    if latents_epoch.shape[0] > max_samples:
        latents_epoch = latents_epoch[np.random.choice(latents_epoch.shape[0], max_samples, replace=False)]

    latents_epoch = latents_epoch - latents_epoch.mean(axis=0)
    pca = PCA()
    X_pca = pca.fit_transform(latents_epoch)

    # Plot RGB-style composite
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], s=6, alpha=0.3, label="PC1 vs PC2", color='red', marker='o')
    ax.scatter(X_pca[:, 1], X_pca[:, 2], s=6, alpha=0.3, label="PC2 vs PC3", color='green', marker='^')
    ax.scatter(X_pca[:, 2], X_pca[:, 3], s=6, alpha=0.3, label="PC3 vs PC4", color='blue', marker='x')

    ax.set_xlim(pca_limits["pc1_2"]["xlim"])
    ax.set_ylim(pca_limits["pc1_2"]["ylim"])
    ax.set_xlabel("PCA Horizontal Axis")
    ax.set_ylabel("PCA Vertical Axis")
    ax.set_title(f"PCA Scatter Overlay — Epoch {suffix}")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    fig.tight_layout()

    # ✅ Final output path
    out_path = os.path.join(prefix_path, f"scatter_combined_{suffix}.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"✅ Saved PCA scatter: {out_path}")


def plot_reconstruction_error(ckpt_paths, save_path="latent_outputs/mean_reconstruction_error.png"):
    train_loss = []
    val_loss = []
    epochs = []

    for path in ckpt_paths:
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            train_buffer = state.get("train_loss_buffer")
            val_buffer = state.get("val_loss_buffer")
            if train_buffer is not None and val_buffer is not None:
                train_loss.append(np.mean(train_buffer))
                val_loss.append(np.mean(val_buffer))
                epochs.append(state.get("epoch", len(epochs) + 1))
        except Exception as e:
            print(f"⚠️ Skipping {path}: {e}")

    if not train_loss:
        print("❌ No reconstruction loss data found.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, label="Train Loss", marker='o')
    ax.plot(epochs, val_loss, label="Val Loss", marker='x')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title("Reconstruction Loss Over Epochs")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"✅ Saved reconstruction error plot to: {save_path}")

    def estimate_gaussian_entropy(latents_window):
        """
        Estimate differential entropy for a multivariate Gaussian over a window.
        Input: latents_window [window_size, latent_dim]
        Returns: scalar entropy estimate
        """
        Z = latents_window - np.mean(latents_window, axis=0)
        cov = np.cov(Z.T)
        sign, logdet = np.linalg.slogdet(cov + 1e-6 * np.eye(Z.shape[1]))  # stabilize
        return 0.5 * logdet + 0.5 * Z.shape[1] * (1 + np.log(2 * np.pi))

    def compute_latent_entropy_per_window(latents, batch_windwo=128, bins=None):
        """
        Compute entropy per 128-snapshot window over all epochs.

        Args:
            latents: ndarray (nepoch, nbatch, latent_dim)
            batch_windwo: int, number of snapshots per window (default: 128)
            bins: unused, kept for API compatibility

        Returns:
            entropy_list: list of tuples (epoch_idx, start_batch_idx, entropy)
        """
        nepoch, nbatch, latent_dim = latents.shape
        entropy_list = []

        for epoch in range(nepoch):
            for start in range(0, nbatch - batch_windwo + 1, batch_windwo):
                window = latents[epoch, start:start + batch_windwo]  # shape [128, latent_dim]
                entropy = estimate_gaussian_entropy(window)
                entropy_list.append((epoch, start, entropy))

        return entropy_list  # list of (epoch_idx, batch_start_idx, entropy)


def main():
    config = load_config("spectral_frontend/configs/basic_config.yaml")

    # --- Resolve checkpoint input ---
    ckpt_dir = config["training_analyzer"]["checkpoint_dir"]
    basename = os.path.basename(ckpt_dir)  # e.g., "250513a_vctk"
    today = datetime.datetime.now().strftime("%y%m%d")  # e.g., "250515"

    # --- Set up per-run output directory ---
    output_dir = os.path.join("training_analysis", basename)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load checkpoints ---
    ckpt_paths = get_checkpoint_paths(ckpt_dir)
    num_epochs = len(ckpt_paths)
    ckpt_path = ckpt_paths[-1]
    print(f"Loading latest checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # extract the latent_dim from checkpoint
    # Try to infer latent_dim from checkpoint shape
    try:
        latents = state["latents"]  # shape: [N, latent_dim]
        inferred_latent_dim = latents.shape[1]
    except Exception:
        # fallback to config
        inferred_latent_dim = config["training"]["latent_dim"]

    # --- Load model from .py in ckpt_dir ---
    model_class = load_model_class_from_checkpoint_dir(ckpt_dir)
    model = model_class(latent_dim=inferred_latent_dim)
    model.load_state_dict(state["model_state_dict"])

    #  Print loaded model
    print("\nLoaded Model Architecture:\n")
    print(model)

    # ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # --------- preload all arrays once from the checkpoins ---------
    recon = load_recon_from_paths(ckpt_paths)  # shape: (nepoch, nbatch, 180)
    y = load_y_from_paths(ckpt_paths)  # shape: (nepoch, nbatch, 180)
    latents = load_latents_from_paths(ckpt_paths)  # shape: (nepoch, nbatch, latent_dim)

    # --------- analyze reconstruction error ------------
    mean_rec_error = config["training_analyzer"].get("mean_reconstruction_error", {})
    if mean_rec_error.get("enable", False):
        png_filename = os.path.join(output_dir, f"reconstruction_error_{today}_{basename}.png")
        plot_reconstruction_error(ckpt_paths, save_path=png_filename)

    # --------- analyze consine similarity
    cosine_cfg = config["training_analyzer"].get("cosine_similarity", {})
    cosine_enabled = cosine_cfg.get("enable", False)
    if cosine_enabled:
        layer_ids = cosine_cfg.get("conv_layers", [])

        for layer in layer_ids:
            print(f"\n Processing conv{layer}")
            weights_over_time = []

            for epoch_idx, ckpt_path in enumerate(ckpt_paths):
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    weights = extract_conv_weights(ckpt, "encoder", layer)
                    weights_over_time.append(weights)
                except Exception as e:
                    print(f"⚠️ Failed to load epoch {epoch_idx}: {ckpt_path}\n{e}")

            if not weights_over_time:
                print(f"❌ No weights found for conv{layer}, skipping.")
                continue

            cosine_sims = compute_cosine_sims(weights_over_time)
            plot_dir = os.path.join("cosine_outputs", f"conv{layer}")
            # plot_epoch_cosines(cosine_sims, plot_dir)
            print(f"✅ Saved cosine similarity plots to: {plot_dir}")

        plot_cosine_summary(cosine_sims, save_path="cosine_outputs/conv2/summary_cosine.png")

    # --------- analyze latent variance ------------
    latent_variance = config["training_analyzer"].get("latent_variance", {})
    if latent_variance.get("enable", False):
        # print("📊 Computing latent variance over epochs...")
        # latents: shape [nepoch, nsnap, latent_dim]
        nepoch, nbatch, latent_dim = latents.shape

        # Compute variance per dim for each epoch
        latent_vars = np.var(latents, axis=1)  # shape: [nepoch, latent_dim]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(
            latent_vars.T, aspect='auto', cmap='cividis', interpolation='nearest'
        )
        cbar = fig.colorbar(im, ax=ax, label="Variance")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Latent Dimension")
        ax.set_title("Latent Variance Over Epochs")
        ax.grid(False)
        plt.tight_layout()

        png_filename = os.path.join(output_dir, f"latent_variance_{today}_{basename}.png")
        fig.savefig(png_filename)
        plt.close(fig)
        print(f"Saved latent variance plot to: {png_filename}")

    # --------- analyze latent entropy ------------
    latent_entropy = config["training_analyzer"].get("latent_entropy", {})
    if latent_entropy.get("enable", False):
        print("📊 Computing latent entropy over epochs...")

        nepoch, nsnap, latent_dim = latents.shape

        # Compute entropy for each epoch (per batch of latent vectors)
        entropy_vec = np.zeros(nepoch, dtype=np.float32)
        for i in range(nepoch):
            Z = latents[i] - latents[i].mean(axis=0)
            cov = np.cov(Z.T)
            _, logdet = np.linalg.slogdet(cov + 1e-6 * np.eye(latent_dim))
            entropy_vec[i] = 0.5 * logdet + 0.5 * latent_dim * (1 + np.log(2 * np.pi))

        # Plot entropy curve over time
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(entropy_vec, marker='o', linestyle='-', alpha=0.8)
        ax.set_title("Latent Entropy Over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Entropy ≈ I(Z;X)")
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()

        png_filename = os.path.join(output_dir, f"latent_entropy_{today}_{basename}.png")
        fig.savefig(png_filename)
        plt.close(fig)
        print(f"✅ Saved latent entropy plot to: {png_filename}")

    # --------- analyze latent PCA ------------
    latent_pca = config["training_analyzer"].get("latent_PCA", {})
    if latent_pca.get("enable", False):
        print("🔎 Performing latent PCA analysis...")
        pca_png_dir = os.path.join(output_dir, "latent_pca_pngs")
        os.makedirs(pca_png_dir, exist_ok=True)
        # Where the final GIFs should go
        gif_dir = output_dir  # one level above png_dir

        # Compute global PCA limits once from all latents
        limits = get_global_pca_limits_from_latents(latents)

        if True:
            # Plot PCA and eigenvalues for each epoch
            for epoch_idx in range(latents.shape[0]):
                latents_epoch = latents[epoch_idx]  # shape: (nbatch, latent_dim)
                suffix = f"{epoch_idx:02d}"

                # Plot eigenvalue bar chart
                plot_latent_pca_eigenvalues_epoch(
                    latents_epoch,
                    prefix_path=pca_png_dir,
                    suffix=suffix
                )
                # Plot 2D PCA scatter projections with consistent limits
                plot_latent_pca_epoch_scatter(
                    latents_epoch,
                    prefix_path=pca_png_dir,
                    suffix=suffix,
                    pca_limits=limits
                )

        # Generate scatter plot GIF
        make_gif_from_pca_series(
            prefix_path=pca_png_dir,
            pattern="scatter_combined_*.png",
            fps=5,
            output_path=os.path.join(gif_dir, "latent_pca_scatter.gif")
        )

        # Generate eigenvalue bar chart GIF
        make_gif_from_pca_series(
            prefix_path=pca_png_dir,
            pattern="eigen_*.png",
            fps=5,
            output_path=os.path.join(gif_dir, "latent_pca_eigen.gif")
        )
        print(f"🎞️ Saved PCA animation GIFs to: {output_dir}")

    ib_analysis = config["training_analyzer"].get("ib_analysis", {})
    if ib_analysis.get("enable", False):

        # Set window size
        batch_window_size = 128
        nepoch, nbatch, in_dim = y.shape
        latent_dim = latents.shape[-1]

        # Clip to multiple of window size
        nbatch_win = nbatch // batch_window_size
        nwin = nepoch * nbatch_win

        # Clip and reshape
        recon_clip = recon[:, :nbatch_win * batch_window_size, :]  # [28, 5632, 180] → [28, 5632, 180]
        y_clip = y[:, :nbatch_win * batch_window_size, :]
        latents_reshape = latents.reshape((nepoch, nbatch, latent_dim))
        latents_clip = latents_reshape[:, :nbatch_win * batch_window_size, :]

        # Final shape: (1232, 128, dim)
        recon_win = recon_clip.reshape((nwin, batch_window_size, in_dim))
        y_win = y_clip.reshape((nwin, batch_window_size, in_dim))
        latents_win = latents_clip.reshape((nwin, batch_window_size, latent_dim))

        # recon_win, y_win: shape (1232, 128, 180)
        ss_res = np.sum((y_win - recon_win) ** 2, axis=(1, 2))  # sum of squared residuals
        y_mean = np.mean(y_win, axis=1, keepdims=True)  # shape: (1232, 1, 180)
        ss_tot = np.sum((y_win - y_mean) ** 2, axis=(1, 2))  # shape: (1232,)
        r2_vec = 1.0 - (ss_res / ss_tot)  # shape: (1232,)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(r2_vec, marker='.', linestyle='-', alpha=0.7, label='R² per window')
        ax.set_title("R² over 128-snapshot windows")
        ax.set_xlabel("Window index (0 to 1231)")
        ax.set_ylabel("R² (I(Z;Y) proxy)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(0, 1)
        ax.legend()
        plt.tight_layout()
        r2_score_png_filename = f"{output_dir}/r2_score_{today}_{basename}.png"
        fig.savefig(r2_score_png_filename)
        plt.close(fig)
        print(f"Saved R2 plot to: {r2_score_png_filename}")

        # calculate entropy over time
        entropy_vec = np.empty((latents_win.shape[0],), dtype=np.float32)

        for i in range(latents_win.shape[0]):
            z = latents_win[i] - np.mean(latents_win[i], axis=0)
            cov = np.cov(z.T)
            _, logdet = np.linalg.slogdet(cov + 1e-6 * np.eye(cov.shape[1]))
            entropy_vec[i] = 0.5 * logdet + 0.5 * cov.shape[0] * (1 + np.log(2 * np.pi))

        r2_vs_entropy_png_filename = f"{output_dir}/r2_vs_entropy_{today}_{basename}.png"
        cmap = plt.get_cmap("viridis")
        colors = np.linspace(0, 1, len(r2_vec))
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(entropy_vec, r2_vec, c=colors, cmap=cmap, alpha=0.6)
        ax.set_xlabel("Entropy (I(Z;X))")
        ax.set_ylabel("R² (I(Z;Y))")
        ax.set_title("IB Curve Colored by Training Time")
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.colorbar(sc, ax=ax, label="Window Index")
        plt.tight_layout()
        r2_vs_entropy_png_filename = f"{output_dir}/r2_vs_entropy_colored_{today}_{basename}.png"
        fig.savefig(r2_vs_entropy_png_filename)
        print(f"Saved R2 vs. entropy plot to: {r2_vs_entropy_png_filename}")


if __name__ == "__main__":
    main()

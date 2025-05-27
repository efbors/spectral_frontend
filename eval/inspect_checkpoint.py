# inspect_checkpoint.py

import torch
import sys
from spectral_frontend.spectral_conv_ae2 import SpectralConvAe2

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/250512b/epoch1.pt"
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# Rebuild model and load weights
model = SpectralConvAe2(latent_dim=128)
model.load_state_dict(state["model_state_dict"])

# Print architecture
print(model)

print(f"Loaded checkpoint: {ckpt_path}")
print("Top-level keys and types:")
for k, v in state.items():
    if isinstance(v, dict):
        print(f"  {k}: dict with {len(v)} keys")
    elif isinstance(v, list):
        print(f"  {k}: list of length {len(v)}")
    elif torch.is_tensor(v):
        print(f"  {k}: tensor shape {tuple(v.shape)}")
    else:
        print(f"  {k}: {type(v)}")

# Optional: inspect scheduler state (if present)
if "scheduler_state_dict" in state and state["scheduler_state_dict"] is not None:
    print("\nScheduler keys:")
    print(state["scheduler_state_dict"].keys())


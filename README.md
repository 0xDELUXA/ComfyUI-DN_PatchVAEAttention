# ComfyUI-DN_PatchVAEAttention

A ComfyUI custom node that patches the attention mechanism inside a VAE's encoder and decoder to use a specific implementation, regardless of what ComfyUI auto-selected at startup.

## Requirements

No additional packages required - all backends are part of ComfyUI's existing dependencies. xformers support requires xformers to be installed in your ComfyUI Python environment.

## Installation

Clone into your ComfyUI custom nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/0xDELUXA/ComfyUI-DN_PatchVAEAttention
```

Then restart ComfyUI.

## Usage

Find the node under **DN > VAEAttention > Patch VAE Attention DN**.

Wire it between your VAE loader and decoder - the patched VAE output is what you connect downstream:

```
Load VAE --> Patch VAE Attention DN --> VAE Decode
```

Or from a checkpoint:

```
Load Checkpoint (VAE) --> Patch VAE Attention DN --> VAE Decode
```

Set `enabled` to `False` to bypass the patch and pass the VAE through unchanged.

## Notes

- ComfyUI automatically selects a VAE attention backend at startup based on your hardware and installed packages; this node lets you override that per-workflow without restarting
- ComfyUI will still print a message like `Using split attention in VAE` or `Using xformers attention in VAE` at startup - this is logged before any node runs and can be ignored; the node is working if `PatchVAEAttentionDN: VAE attention successfully patched to X` appears in the log shortly after
- If xformers is unavailable, the node automatically falls back to PyTorch attention and logs a warning
- Tested with SD1.5, SDXL, and Flux VAEs; should also work with other models

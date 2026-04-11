import logging

import comfy.model_management as mm
from comfy.ldm.modules.diffusionmodules.model import AttnBlock


# Attention implementations

def _pytorch_attention(q, k, v):
    import comfy.ldm.modules.diffusionmodules.model as vae_model
    return vae_model.pytorch_attention(q, k, v)


def _xformers_attention(q, k, v):
    import comfy.ldm.modules.diffusionmodules.model as vae_model
    if not mm.xformers_enabled_vae():
        logging.warning("PatchVAEAttentionDN: xformers not available for VAE, falling back to pytorch")
        return vae_model.pytorch_attention(q, k, v)
    return vae_model.xformers_attention(q, k, v)


def _slice_attention(q, k, v):
    import comfy.ldm.modules.diffusionmodules.model as vae_model
    return vae_model.normal_attention(q, k, v)


_BACKENDS = {
    "pytorch":  _pytorch_attention,
    "xformers": _xformers_attention,
    "split":    _slice_attention,
}


# Patch helpers

def _patch_vae(vae, attn_fn):
    model = vae.first_stage_model
    patched = 0
    for module in model.modules():
        if isinstance(module, AttnBlock):
            module.optimized_attention = attn_fn
            patched += 1
    return vae


# ComfyUI node

class PatchVAEAttentionDN:
    """
    Patches the attention implementation used inside a VAE's encoder/decoder.
    ComfyUI automatically selects a VAE attention backend at startup,
    use this node to override that selection per-workflow.

    Set enabled to False to bypass the patch and pass the VAE through unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "attention": (list(_BACKENDS.keys()),),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "patch"
    DESCRIPTION = (
        "Patches the attention implementation used inside a VAE's encoder and decoder. "
        "Overrides ComfyUI's auto-selected backend per-workflow without restarting. "
        "Wire the output VAE into your decoder."
    )
    EXPERIMENTAL = True
    CATEGORY = "DN/VAEAttention"

    def patch(self, vae, attention: str, enabled: bool):
        if not enabled:
            return (vae,)

        attn_fn = _BACKENDS[attention]
        _patch_vae(vae, attn_fn)
        logging.info(f"PatchVAEAttentionDN: VAE attention successfully patched to {attention}")
        return (vae,)

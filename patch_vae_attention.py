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


def flash_attention(q, k, v):
    from flash_attn import flash_attn_func
    orig_shape = q.shape
    B = orig_shape[0]
    C = orig_shape[1]
    N = orig_shape[2] if len(orig_shape) == 3 else orig_shape[2] * orig_shape[3]
    oom_fallback = False
    q_reshaped = q.view(B, C, N).transpose(1, 2).unsqueeze(2).contiguous()
    k_reshaped = k.view(B, C, N).transpose(1, 2).unsqueeze(2).contiguous()
    v_reshaped = v.view(B, C, N).transpose(1, 2).unsqueeze(2).contiguous()

    try:
        out = flash_attn_func(q_reshaped, k_reshaped, v_reshaped, dropout_p=0.0, causal=False)
        out = out.squeeze(2).transpose(1, 2).reshape(orig_shape)
    except Exception as e:
        mm.raise_non_oom(e)
        logging.warning("OOMed: switched to slice attention")
        oom_fallback = True
    if oom_fallback:
        q_flat = q.view(B, C, N).transpose(1, 2)
        k_flat = k.view(B, C, N)
        v_flat = v.view(B, C, N)
        out = _slice_attention(q_flat, k_flat, v_flat).reshape(orig_shape)
    return out


_BACKENDS = {
    "pytorch":  _pytorch_attention,
    "xformers": _xformers_attention,
    "split":    _slice_attention,
    "flash":    flash_attention,
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

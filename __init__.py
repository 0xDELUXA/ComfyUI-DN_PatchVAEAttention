from .patch_vae_attention import PatchVAEAttentionDN

NODE_CLASS_MAPPINGS = {
    "PatchVAEAttentionDN": PatchVAEAttentionDN,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PatchVAEAttentionDN": "Patch VAE Attention DN",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

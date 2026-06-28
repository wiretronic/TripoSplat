from pathlib import Path
from huggingface_hub import snapshot_download

def get_weights(ckpt_dir="ckpts"):
    snapshot_download(
        repo_id="VAST-AI/TripoSplat",
        local_dir=ckpt_dir,
        allow_patterns=[
            "diffusion_models/*",
            "vae/*",
            "clip_vision/*",
        ],
    )
    d = Path(ckpt_dir)
    return dict(
        ckpt_path=str(d / "diffusion_models/triposplat_fp16.safetensors"),
        decoder_path=str(d / "vae/triposplat_vae_decoder_fp16.safetensors"),
        flux2_vae_encoder_path=str(d / "vae/flux2-vae.safetensors"),
        dinov3_path=str(d / "clip_vision/dino_v3_vit_h.safetensors"),
    )
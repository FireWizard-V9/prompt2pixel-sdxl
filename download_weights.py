import os
from dotenv import load_dotenv
from huggingface_hub import login, hf_hub_download
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

cache_dir = "/mnt/fast-scratch/sdxl-data/hf_cache"
os.makedirs(cache_dir, exist_ok=True)

print("--> Verifying/Downloading FP16 VAE fix...")
AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", 
    cache_dir=cache_dir
)

print("--> Verifying/Downloading Base SDXL 1.0 Pipeline...")
StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    cache_dir=cache_dir
)

print("--> Downloading SDXL-Lightning 4-step LoRA checkpoint...")
hf_hub_download(
    repo_id="ByteDance/SDXL-Lightning",
    filename="sdxl_lightning_4step_lora.safetensors",
    cache_dir=cache_dir
)

print("--> All model weights verified and ready!")
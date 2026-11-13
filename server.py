import os
import io
import uuid
import random
import torch
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_KEY")
S3_BUCKET_NAME = "viv-image-gen-presentation"
S3_REGION = "us-east-1"

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=S3_REGION
)

CACHE_DIR = "/data/hf_cache"
BASE_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
LORA_FILE = "sdxl_lightning_4step_lora.safetensors"

app = FastAPI(title="SDXL-Lightning Production API Service")
pipeline = None

def load_pipeline():
    global pipeline
    print("--> Loading Base SDXL Pipeline with FP16 VAE...")
    
    # 1. Load FP16 VAE Fix to eliminate black image generation bugs
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )
    
    # 2. Load Base SDXL 1.0 Model
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_REPO,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=CACHE_DIR
    ).to("cuda")
    
    # 3. Apply Euler Discrete Scheduler required by SDXL-Lightning
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, 
        timestep_spacing="trailing"
    )
    
    # 4. Attach & Fuse SDXL-Lightning 4-step LoRA weights
    print("--> Loading and fusing SDXL-Lightning 4-step LoRA...")
    lora_path = hf_hub_download(
        repo_id=LIGHTNING_REPO, 
        filename=LORA_FILE, 
        cache_dir=CACHE_DIR
    )
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora()
    
    # 5. Hardware & VRAM Optimizations for NVIDIA T4 GPU
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
    
    # 6. Warmup Inference pass to trigger PyTorch Graph Compilation
    print("--> Performing GPU warmup inference...")
    generator = torch.Generator("cuda").manual_seed(42)
    _ = pipe("a photo of an astronaut on Mars", num_inference_steps=4, guidance_scale=0.0, generator=generator)
    pipeline = pipe
    print("--> SERVICE IS ONLINE AND READY!")

@app.on_event("startup")
def startup_event():
    load_pipeline()

# --- Request Data Models ---
class SingleImageRequest(BaseModel):
    prompt: str
    steps: Optional[int] = 4
    seed: Optional[int] = None

class BatchImageRequest(BaseModel):
    prompt: str
    num_images: Optional[int] = 6
    steps: Optional[int] = 4
    seed: Optional[int] = None

class MultiPromptRequest(BaseModel):
    prompts: List[str]
    steps: Optional[int] = 4

# --- Response Data Models ---
class ImageResponse(BaseModel):
    request_id: str
    prompt: str
    s3_key: str
    seed_used: int
    url: str

class GenerationResponse(BaseModel):
    request_id: str
    count: int
    images: List[ImageResponse]

# --- Shared Generation Routine ---
def process_generation_tasks(tasks: List[tuple], steps: int) -> GenerationResponse:
    request_id = str(uuid.uuid4())
    generated_images = []
    
    for idx, (prompt_text, seed_val) in enumerate(tasks):
        generator = torch.Generator("cuda").manual_seed(seed_val)
        
        # GPU Inference Execution
        img = pipeline(
            prompt=prompt_text,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator
        ).images[0]
        
        # In-Memory PNG Encoding
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Upload directly to AWS S3 without writing to host disk
        s3_key = f"requests/{request_id}/image_{idx+1}.png"
        s3_client.upload_fileobj(
            buffer,
            S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={"ContentType": "image/png"}
        )
        
        # Generate 1-Hour AWS Presigned URL
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": s3_key},
            ExpiresIn=3600
        )
        
        generated_images.append(
            ImageResponse(
                request_id=request_id,
                prompt=prompt_text,
                s3_key=s3_key,
                seed_used=seed_val,
                url=url
            )
        )
        
    return GenerationResponse(
        request_id=request_id,
        count=len(generated_images),
        images=generated_images
    )

# --- Endpoint 1: Single Image ---
@app.post("/generate/single", response_model=GenerationResponse)
def generate_single(req: SingleImageRequest):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline initialisation failure")
    
    seed_val = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
    return process_generation_tasks([(req.prompt, seed_val)], req.steps)

# --- Endpoint 2: Batch Variations ---
@app.post("/generate/batch", response_model=GenerationResponse)
def generate_batch(req: BatchImageRequest):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline initialisation failure")
    
    base_seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
    tasks = [(req.prompt, base_seed + i) for i in range(req.num_images)]
    return process_generation_tasks(tasks, req.steps)

# --- Endpoint 3: Multi-Prompt / Multi-Style ---
@app.post("/generate/multi-prompt", response_model=GenerationResponse)
def generate_multi_prompt(req: MultiPromptRequest):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline initialisation failure")
    
    if not req.prompts:
        raise HTTPException(status_code=400, detail="The 'prompts' array cannot be empty.")
    
    tasks = [(p, random.randint(0, 2**32 - 1)) for p in req.prompts]
    return process_generation_tasks(tasks, req.steps)
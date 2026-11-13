#!/bin/bash
set -euxo pipefail
exec > /var/log/sdxl-bootstrap.log 2>&1

echo "================================================================"
echo " SDXL Bootstrap Started: $(date)"
echo "================================================================"

# ── 1. System update ──────────────────────────────────────────────────────────
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get upgrade -y
apt-get install -y \
    curl wget git unzip jq \
    build-essential dkms \
    linux-headers-$(uname -r) \
    ca-certificates gnupg lsb-release \
    nvme-cli

# ── 2. NVIDIA Driver (if not already present via Deep Learning AMI) ───────────
if ! command -v nvidia-smi &>/dev/null; then
    echo "--> Installing NVIDIA driver 535..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update -y
    apt-get install -y cuda-drivers-535
    echo "--> NVIDIA driver installed. Reboot may be needed — continuing..."
else
    echo "--> nvidia-smi already present: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
fi

# ── 3. Docker ─────────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "--> Installing Docker..."
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
    systemctl enable docker
    systemctl start docker
    usermod -aG docker ubuntu
else
    echo "--> Docker already installed: $(docker --version)"
fi

# ── 4. NVIDIA Container Toolkit ───────────────────────────────────────────────
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "--> Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -y
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    echo "--> NVIDIA Container Toolkit installed."
else
    echo "--> NVIDIA Container Toolkit already installed."
fi

# ── 5. Mount NVMe scratch disk ────────────────────────────────────────────────
echo "--> Setting up NVMe scratch disk..."
SCRATCH_MOUNT="/mnt/fast-scratch"
mkdir -p $SCRATCH_MOUNT

# Find the attached EBS volume (not root)
SCRATCH_DEV=$(lsblk -dpno NAME,SIZE | awk '$2 ~ /[0-9]+G/ {print $1}' | grep -v "$(lsblk -no PKNAME $(findmnt -n -o SOURCE /))" | head -1)

if [ -n "$SCRATCH_DEV" ] && [ -b "$SCRATCH_DEV" ]; then
    if ! blkid "$SCRATCH_DEV" | grep -q ext4; then
        echo "--> Formatting $SCRATCH_DEV as ext4..."
        mkfs.ext4 -F "$SCRATCH_DEV"
    fi
    mount "$SCRATCH_DEV" "$SCRATCH_MOUNT" || true
    echo "$SCRATCH_DEV $SCRATCH_MOUNT ext4 defaults,nofail 0 2" >> /etc/fstab
    echo "--> Mounted $SCRATCH_DEV at $SCRATCH_MOUNT"
else
    echo "--> No secondary EBS found, using root volume for scratch."
fi

mkdir -p $SCRATCH_MOUNT/sdxl-data/hf_cache
chown -R ubuntu:ubuntu $SCRATCH_MOUNT

# ── 6. Copy project files ─────────────────────────────────────────────────────
echo "--> Writing project files..."
APP_DIR="/home/ubuntu/sdxl-tensorrt-service"
mkdir -p $APP_DIR/prompt_enhancer

# --- server.py ---
cat > $APP_DIR/server.py << 'SERVERPY'
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
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_REPO,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=CACHE_DIR
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing"
    )
    print("--> Loading and fusing SDXL-Lightning 4-step LoRA...")
    lora_path = hf_hub_download(
        repo_id=LIGHTNING_REPO,
        filename=LORA_FILE,
        cache_dir=CACHE_DIR
    )
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora()
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
    print("--> Performing GPU warmup inference...")
    generator = torch.Generator("cuda").manual_seed(42)
    _ = pipe("a photo of an astronaut on Mars", num_inference_steps=4, guidance_scale=0.0, generator=generator)
    pipeline = pipe
    print("--> SERVICE IS ONLINE AND READY!")

@app.on_event("startup")
def startup_event():
    load_pipeline()

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

def process_generation_tasks(tasks: List[tuple], steps: int) -> GenerationResponse:
    request_id = str(uuid.uuid4())
    generated_images = []
    for idx, (prompt_text, seed_val) in enumerate(tasks):
        generator = torch.Generator("cuda").manual_seed(seed_val)
        img = pipeline(
            prompt=prompt_text,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator
        ).images[0]
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        s3_key = f"requests/{request_id}/image_{idx+1}.png"
        s3_client.upload_fileobj(buffer, S3_BUCKET_NAME, s3_key, ExtraArgs={"ContentType": "image/png"})
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": s3_key},
            ExpiresIn=3600
        )
        generated_images.append(ImageResponse(request_id=request_id, prompt=prompt_text, s3_key=s3_key, seed_used=seed_val, url=url))
    return GenerationResponse(request_id=request_id, count=len(generated_images), images=generated_images)

@app.post("/generate/single", response_model=GenerationResponse)
def generate_single(req: SingleImageRequest):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline initialisation failure")
    seed_val = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
    return process_generation_tasks([(req.prompt, seed_val)], req.steps)

@app.post("/generate/batch", response_model=GenerationResponse)
def generate_batch(req: BatchImageRequest):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline initialisation failure")
    base_seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
    tasks = [(req.prompt, base_seed + i) for i in range(req.num_images)]
    return process_generation_tasks(tasks, req.steps)

@app.post("/generate/multi-prompt", response_model=GenerationResponse)
def generate_multi_prompt(req: MultiPromptRequest):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline initialisation failure")
    if not req.prompts:
        raise HTTPException(status_code=400, detail="The 'prompts' array cannot be empty.")
    tasks = [(p, random.randint(0, 2**32 - 1)) for p in req.prompts]
    return process_generation_tasks(tasks, req.steps)
SERVERPY

# --- requirements.txt (GPU service) ---
cat > $APP_DIR/requirements.txt << 'REQS'

torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
transformers==4.38.2
diffusers==0.26.3
safetensors==0.4.2
huggingface-hub<0.25.0
numpy<2.0.0
accelerate==0.27.2
fastapi==0.110.0
uvicorn==0.28.0
pillow==10.2.0
boto3==1.34.50
python-dotenv==1.0.1
REQS

# --- Dockerfile (GPU service) ---
cat > $APP_DIR/Dockerfile << 'DOCKERGPU'
FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
DOCKERGPU

# --- download_weights.py ---
cat > $APP_DIR/download_weights.py << 'DLWEIGHTS'
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
AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", cache_dir=cache_dir)

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
DLWEIGHTS

# --- .env (GPU service) ---
cat > $APP_DIR/.env << ENVGPU
AWS_ACCESS_KEY_ID=${aws_access_key_id}
AWS_SECRET_KEY=${aws_secret_access_key}
HF_TOKEN=${hf_token}
ENVGPU

# --- prompt_enhancer/agent.py ---
cat > $APP_DIR/prompt_enhancer/agent.py << 'AGENTPY'
import os
import uuid
import datetime
import requests
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SDXL_SERVICE_URL = os.getenv("SDXL_SERVICE_URL", "http://localhost:8000")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY environment variable is missing.")

def generate_timestamped_trace_id() -> str:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{timestamp}-{short_uuid}"

def truncate_to_clip_limit(prompt_text: str, max_words: int = 60) -> str:
    words = prompt_text.strip().split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return prompt_text

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://sdxl-agent.internal",
        "X-Title": "LangChain SDXL Prompt Enhancer Agent",
    },
    temperature=0.6
)

class EnhancedPromptOutput(BaseModel):
    enhanced_prompts: List[str] = Field(..., description="List of concise SDXL prompts under 60 words.")

    @field_validator("enhanced_prompts")
    @classmethod
    def validate_and_truncate_prompts(cls, prompts: List[str]) -> List[str]:
        return [" ".join(p.strip().split()[:60]) if len(p.strip().split()) > 60 else p.strip() for p in prompts]

structured_llm = llm.with_structured_output(EnhancedPromptOutput)

SYSTEM_PROMPT = """
You are 'VisionCrafter-Agent', an elite Diffusion Prompt Engineer.
Your goal is to convert user concepts into concise, visual SDXL 1.0 prompts.

System Time Context: {current_datetime}

### HARD CONSTRAINTS:
1. Every generated prompt MUST BE BETWEEN 35 AND 55 WORDS (STRICT MAXIMUM 60 WORDS).
2. If the user specifies an explicit style, RESPECT IT STRICTLY.
3. NO BUZZWORDS: Avoid "masterpiece", "best quality", "hyperrealistic".

### CONCISE FEW-SHOT EXAMPLES:

Example 1 (Chemistry):
User Concept: "Electrolysis of water experiment setup"
Expanded Prompt: "Scientific laboratory photo of a glass Hoffmann voltameter filled with acidified water. Platinum electrodes producing oxygen and hydrogen bubbles at anode and cathode. Clear volumetric marks, lab bench background, soft ring-light illumination, crisp reflections, textbook 3D render quality."

Example 2 (History):
User Concept: "Emperor Ashoka after Kalinga war"
Expanded Prompt: "Dramatic historical oil painting of Emperor Ashoka on the Kalinga battlefield, somber facial expression looking down at his sword. Wearing Mauryan silk sashes and weathered bronze armor. Misty battlefield sunset, distant Ashoka pillar, warm amber side-lighting, expressive brushwork."

Example 3 (Biology):
User Concept: "Structure of human neuron firing signal"
Expanded Prompt: "High-resolution 3D medical illustration of a human neuron. Bright glowing blue action potential travelling down the myelin-sheathed axon. Surrounding astrocytes, dark violet background, bioluminescent neurotransmitters releasing at synaptic cleft, depth of field highlighting cell soma."
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "Generate {count} concise prompt variation(s) (MAX 50 WORDS EACH) for: '{user_concept}'")
])

agent_chain = prompt_template | structured_llm

app = FastAPI(title="LangChain SDXL Prompt Enhancer Agent Service")

class AgentUserRequest(BaseModel):
    prompt: str = Field(..., description="Raw user query")
    mode: Optional[str] = Field("single", description="Mode: 'single', 'batch', or 'multi-prompt'")
    num_images: Optional[int] = Field(1, description="Number of images/variations")
    steps: Optional[int] = Field(4, description="Inference steps for SDXL")

@app.post("/agent/process")
def process_agent_request(req: AgentUserRequest, x_request_id: Optional[str] = Header(None)):
    trace_id = x_request_id if x_request_id else generate_timestamped_trace_id()
    count = req.num_images if req.mode == "multi-prompt" else 1
    current_dt_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    try:
        chain_output: EnhancedPromptOutput = agent_chain.invoke({
            "current_datetime": current_dt_str,
            "user_concept": req.prompt,
            "count": count
        })
        enhanced_prompts = [truncate_to_clip_limit(p, max_words=60) for p in chain_output.enhanced_prompts]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LangChain Agent Processing Error: {str(e)}")

    if req.mode == "single":
        target_endpoint = f"{SDXL_SERVICE_URL}/generate/single"
        payload = {"prompt": enhanced_prompts[0], "steps": req.steps}
    elif req.mode == "batch":
        target_endpoint = f"{SDXL_SERVICE_URL}/generate/batch"
        payload = {"prompt": enhanced_prompts[0], "num_images": req.num_images, "steps": req.steps}
    elif req.mode == "multi-prompt":
        target_endpoint = f"{SDXL_SERVICE_URL}/generate/multi-prompt"
        payload = {"prompts": enhanced_prompts, "steps": req.steps}
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'single', 'batch', or 'multi-prompt'.")

    try:
        gpu_response = requests.post(
            target_endpoint,
            json=payload,
            headers={"Content-Type": "application/json", "X-Request-ID": trace_id},
            timeout=180
        )
        gpu_response.raise_for_status()
        gpu_result = gpu_response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"GPU Service Inference Failure: {str(e)}")

    return {
        "request_id": trace_id,
        "original_prompt": req.prompt,
        "mode": req.mode,
        "enhanced_prompts_used": enhanced_prompts,
        "gpu_service_response": gpu_result
    }
AGENTPY

# --- prompt_enhancer/requirements.txt ---
cat > $APP_DIR/prompt_enhancer/requirements.txt << 'AGENTREQS'
fastapi
uvicorn
requests
langchain
langchain-openai
pydantic
python-dotenv
AGENTREQS

# --- prompt_enhancer/Dockerfile ---
cat > $APP_DIR/prompt_enhancer/Dockerfile << 'DOCKERAGENT'
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py .

EXPOSE 8001

CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8001"]
DOCKERAGENT

# --- prompt_enhancer/.env ---
# SDXL_SERVICE_URL uses the EC2 private IP (localhost works since both containers are on same host)
cat > $APP_DIR/prompt_enhancer/.env << ENVAGENT
OPENROUTER_API_KEY=${openrouter_api_key}
SDXL_SERVICE_URL=http://localhost:8000
ENVAGENT

chown -R ubuntu:ubuntu $APP_DIR
echo "--> All project files written."

# ── 7. Download model weights (runs as ubuntu user) ───────────────────────────
echo "--> Pre-fetching model weights from HuggingFace Hub..."
cd $APP_DIR

# Install minimal Python deps on host just for the download script
pip3 install --quiet huggingface-hub diffusers python-dotenv torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || true

sudo -u ubuntu HF_TOKEN="${hf_token}" python3 $APP_DIR/download_weights.py \
    >> /var/log/sdxl-bootstrap.log 2>&1 && \
    echo "--> Model weights downloaded successfully." || \
    echo "--> WARNING: Weight download failed — will retry inside container on first start."

# ── 8. Build Docker images ────────────────────────────────────────────────────
echo "--> Building GPU inference Docker image (sdxl-s3-service:v1)..."
docker build -t sdxl-s3-service:v1 $APP_DIR \
    >> /var/log/sdxl-bootstrap.log 2>&1
echo "--> GPU image built."

echo "--> Building Prompt Enhancer Agent Docker image (sdxl-agent:v1)..."
docker build -t sdxl-agent:v1 $APP_DIR/prompt_enhancer \
    >> /var/log/sdxl-bootstrap.log 2>&1
echo "--> Agent image built."

# ── 9. Start containers ───────────────────────────────────────────────────────
echo "--> Starting GPU inference container (Port 8000)..."
docker run -d \
    --name sdxl_service \
    --restart always \
    --gpus all \
    -p 8000:8000 \
    -v /mnt/fast-scratch/sdxl-data:/data \
    --env-file $APP_DIR/.env \
    sdxl-s3-service:v1

echo "--> Starting Prompt Enhancer Agent container (Port 8001)..."
docker run -d \
    --name sdxl_agent \
    --restart always \
    -p 8001:8001 \
    --env-file $APP_DIR/prompt_enhancer/.env \
    sdxl-agent:v1

# ── 10. Health check & smoke test ────────────────────────────────────────────
echo "--> Waiting for GPU service to come online (pipeline warmup ~3-5 min)..."
MAX_WAIT=600   # 10 minutes max
ELAPSED=0
until curl -sf http://localhost:8000/docs > /dev/null 2>&1; do
    sleep 15
    ELAPSED=$((ELAPSED + 15))
    echo "    ... waiting ${ELAPSED}s / ${MAX_WAIT}s"
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "--> WARNING: GPU service did not respond within ${MAX_WAIT}s. Check: docker logs sdxl_service"
        break
    fi
done

if curl -sf http://localhost:8000/docs > /dev/null 2>&1; then
    echo "--> GPU service is UP at http://localhost:8000"
    echo "--> Running smoke test: POST /generate/single..."
    SMOKE=$(curl -sf -X POST http://localhost:8000/generate/single \
        -H "Content-Type: application/json" \
        -d '{"prompt": "a red apple on a white table, studio lighting", "steps": 4}' || echo "FAILED")
    echo "--> Smoke test result: $SMOKE"
fi

echo "--> Waiting for Agent service..."
sleep 5
if curl -sf http://localhost:8001/docs > /dev/null 2>&1; then
    echo "--> Agent service is UP at http://localhost:8001"
fi

echo "================================================================"
echo " SDXL Bootstrap Complete: $(date)"
echo " GPU Service  → http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
echo " Agent Service → http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8001"
echo " Logs: /var/log/sdxl-bootstrap.log"
echo " Container logs: docker logs sdxl_service | docker logs sdxl_agent"
echo "================================================================"

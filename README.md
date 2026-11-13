# 🎨 SDXL Lightning — GPU Image Generation Service

> **Fully automated, one-command deployment** of a production-grade AI image generation system on AWS `g4dn.xlarge` (NVIDIA Tesla T4). Terraform provisions everything — VPC, EC2, storage, NVIDIA drivers, Docker containers, model weights — and starts both microservices automatically.

---

## 📐 System Architecture

```
                        ┌─────────────────────────────────────────────────────────────┐
                        │                  ☁️  AWS Cloud  —  us-east-1                │
                        │                                                             │
                        │   ┌─────────────────────────────────────────────────────┐  │
                        │   │            🔒 VPC  10.0.0.0/16                      │  │
                        │   │                                                     │  │
                        │   │   ┌─────────────────────────────────────────────┐   │  │
                        │   │   │       📡 Public Subnet  10.0.1.0/24         │   │  │
                        │   │   │                                             │   │  │
                        │   │   │   🛡️ Security Group  TCP 22 · 8000 · 8001   │   │  │
                        │   │   │                                             │   │  │
                        │   │   │   ┌─────────────────────────────────────┐   │   │  │
                        │   │   │   │      ⚡ EC2  g4dn.xlarge            │   │   │  │
                        │   │   │   │                                     │   │   │  │
                        │   │   │   │  ┌──────────────────────────────┐   │   │   │  │
                        │   │   │   │  │     🐳 Docker Runtime        │   │   │   │  │
  🖥️  Client            │   │   │   │  │                              │   │   │   │  │
      │                 │   │   │   │  │  🤖 sdxl-agent:v1  :8001    │   │   │   │  │
      │ POST            │   │   │   │  │     python:3.10-slim         │   │   │   │  │
      │ /agent/process  │   │   │   │  │     LangChain + FastAPI      │   │   │   │  │
      │ :8001           │   │   │   │  │            │                 │   │   │   │  │
      │                 │   │   │   │  │            │ POST            │   │   │   │  │
      │                 │   │   │   │  │            │ /generate/{mode}│   │   │   │  │
      │                 │   │   │   │  │            ▼                 │   │   │   │  │
      │                 │   │   │   │  │  🔥 sdxl-s3-service:v1 :8000│   │   │   │  │
      │                 │   │   │   │  │     nvidia/pytorch:23.10-py3 │   │   │   │  │
      │                 │   │   │   │  │     SDXL-Lightning Pipeline  │   │   │   │  │
      │                 │   │   │   │  └──────────────┬───────────────┘   │   │   │  │
      │                 │   │   │   │                 │                   │   │   │  │
      │                 │   │   │   │  🎮 NVIDIA Tesla T4  16GB VRAM      │   │   │  │
      │                 │   │   │   │  💾 EBS 100GB  /mnt/fast-scratch    │   │   │  │
      │                 │   │   │   └─────────────────┼───────────────────┘   │   │  │
      │                 │   │   └───────────────────── │ ─────────────────────┘   │  │
      │                 │   └─────────────────────────  │ ───────────────────────┘  │  │
      │                 │                               │ upload_fileobj()           │  │
      │                 │   🪣 S3: viv-image-gen-presentation ◄──────────────────   │  │
      │                 │      requests/{id}/image_N.png                            │  │
      │                 └─────────────────────────────────────────────────────────┘  │
      │                                                                              │
      │  ◄──────────────────────────── Presigned URL (1hr TTL) ─────────────────────┘
      │
      │  🧠 OpenRouter (gpt-4o-mini) ◄── agent.py LangChain invoke
      │  🤗 HuggingFace Hub ──► download_weights.py (~15GB, first boot only)
```

---

## 🧠 AI Model Pipeline

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    SDXL Inference Pipeline                          │
  │                                                                     │
  │   📦 stabilityai/stable-diffusion-xl-base-1.0                      │
  │      FP16 UNet  ·  CLIP ViT-L  ·  OpenCLIP ViT-bigG                │
  │      1024 × 1024 native resolution                                  │
  │                          │                                          │
  │              pipe.load_lora_weights()                               │
  │                          ▼                                          │
  │   ⚡ ByteDance/SDXL-Lightning                                       │
  │      sdxl_lightning_4step_lora.safetensors                          │
  │      Progressive Adversarial Distillation  →  20-50 steps to 4     │
  │                          │                                          │
  │                  pipe.fuse_lora()                                   │
  │              (baked into UNet weights)                              │
  │                          │                                          │
  │            ┌─────────────┴──────────────┐                          │
  │            ▼                            ▼                          │
  │   🔧 madebyollin/sdxl-vae-fp16-fix   🕐 EulerDiscreteScheduler     │
  │      FP16-safe VAE decoder               timestep_spacing=trailing  │
  │      Prevents black images on T4         guidance_scale=0.0         │
  │            │                            │                          │
  │            └─────────────┬──────────────┘                          │
  │                          ▼                                          │
  │   🚀 torch.compile(unet, mode="reduce-overhead")                   │
  │      Fused CUDA kernels  ·  ~2s per image after warmup             │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Request Lifecycle

```
  🖥️  Client
      │
      │  1.  POST /agent/process  :8001
      │      { "prompt": "...", "mode": "batch", "num_images": 3, "steps": 4 }
      ▼
  ┌─────────────────────────────────────────────────────┐
  │  🤖  agent.py  (Port 8001)                          │
  │                                                     │
  │  ① generate_timestamped_trace_id()                  │
  │     → "20261109-045412-a1b2c3d4"                    │
  │                                                     │
  │  ② agent_chain.invoke()                             │
  │     LangChain → OpenRouter → gpt-4o-mini            │
  │     structured output → EnhancedPromptOutput        │
  │     Pydantic validated · CLIP-truncated ≤ 60 words  │
  └──────────────────────┬──────────────────────────────┘
                         │
                         │  2.  POST /generate/batch  :8000
                         │      Header: X-Request-ID: 20261109-045412-a1b2c3d4
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │  🔥  server.py  (Port 8000)                         │
  │                                                     │
  │  for i in range(num_images):                        │
  │    seed = base_seed + i                             │
  │    ③ pipeline(prompt, steps=4, guidance_scale=0.0)  │
  │       → PIL Image [1024 × 1024]  on NVIDIA T4       │
  │    ④ img.save(io.BytesIO(), "PNG")                  │
  │       zero disk I/O                                 │
  │    ⑤ s3_client.upload_fileobj()                     │
  │       → requests/{id}/image_{i+1}.png               │
  │    ⑥ generate_presigned_url(ExpiresIn=3600)         │
  └──────────────────────┬──────────────────────────────┘
                         │
                         │  3.  GenerationResponse JSON
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │  🤖  agent.py  wraps response                       │
  │      { request_id, original_prompt,                 │
  │        enhanced_prompts_used,                       │
  │        gpu_service_response }                       │
  └──────────────────────┬──────────────────────────────┘
                         │
                         │  4.  Final JSON response
                         ▼
  🖥️  Client  ←  presigned S3 URL valid for 1 hour
```

---

## 🏗️ Terraform — What Gets Created

```
  terraform apply
       │
       ├── 🌐 Networking
       │     aws_vpc                  10.0.0.0/16
       │     aws_internet_gateway
       │     aws_subnet               10.0.1.0/24  us-east-1a
       │     aws_route_table          0.0.0.0/0 → IGW
       │     aws_route_table_association
       │
       ├── 🔒 Security
       │     aws_security_group       TCP 22 · 8000 · 8001
       │     aws_key_pair             ~/.ssh/id_rsa.pub
       │
       ├── 💾 Storage
       │     aws_ebs_volume           100 GB gp3  →  /mnt/fast-scratch
       │     aws_volume_attachment    /dev/sdf
       │
       └── ⚡ Compute
             aws_instance             g4dn.xlarge · Deep Learning AMI · 80GB root gp3
                  │
                  └── user_data: userdata.sh.tpl  (runs automatically on first boot)
                            │
                            ├── ①  apt update · build-essential · nvme-cli
                            ├── ②  NVIDIA Driver 535  (skip if DL AMI has it)
                            ├── ③  Docker CE  +  systemctl enable docker
                            ├── ④  NVIDIA Container Toolkit  +  nvidia-ctk configure
                            ├── ⑤  Format + mount EBS  →  /mnt/fast-scratch  ext4
                            ├── ⑥  Write all project files  (secrets injected from tfvars)
                            │        server.py · agent.py · Dockerfiles · .env files
                            ├── ⑦  python3 download_weights.py  (~15 GB to NVMe)
                            ├── ⑧  docker build sdxl-s3-service:v1
                            │        docker build sdxl-agent:v1
                            ├── ⑨  docker run --gpus all --restart always  :8000
                            │        docker run --restart always             :8001
                            └── ⑩  Health check loop + smoke test POST /generate/single
```

---

## 📁 Project Structure

```
Image_generation/
│
├── 📄 server.py                 GPU inference engine          Port 8000
├── 📄 Dockerfile                nvcr.io/nvidia/pytorch:23.10-py3
├── 📄 requirements.txt          torch · diffusers · boto3 · fastapi (pinned)
├── 📄 download_weights.py       pre-fetches ~15 GB to /mnt/fast-scratch
├── 📄 .env                      AWS_ACCESS_KEY_ID · AWS_SECRET_KEY · HF_TOKEN
│
├── prompt_enhancer/
│   ├── 📄 agent.py              LangChain prompt enhancer     Port 8001
│   ├── 📄 Dockerfile            python:3.10-slim
│   ├── 📄 requirements.txt      langchain · fastapi · openai
│   └── 📄 .env                  OPENROUTER_API_KEY · SDXL_SERVICE_URL
│
├── terraform/
│   ├── 📄 main.tf               VPC · SG · EC2 · EBS · Key Pair
│   ├── 📄 variables.tf          all input variables with defaults
│   ├── 📄 outputs.tf            public IP · SSH cmd · service URLs
│   ├── 📄 terraform.tfvars      ← YOUR SECRETS GO HERE
│   └── 📄 userdata.sh.tpl       full 10-step EC2 bootstrap script
│
├── 📄 README.md
├── 📄 DEPLOY.md                 quick command reference
├── 📄 information.md            deep technical reference
└── 📄 .gitignore                excludes .env · tfvars · tfstate
```

---

## 🔑 Environment Variables

### Keys you need before deploying

| Key | Used By | Where to Get |
|-----|---------|-------------|
| `AWS_ACCESS_KEY_ID` | GPU container → S3 uploads | AWS Console → IAM → Security Credentials |
| `AWS_SECRET_ACCESS_KEY` | GPU container → S3 uploads | Same — shown once on creation |
| `OPENROUTER_API_KEY` | Agent → GPT-4o-Mini | [openrouter.ai/keys](https://openrouter.ai/keys) |
| `HF_TOKEN` | Weight download (optional) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

### The only file you edit — `terraform/terraform.tfvars`

```hcl
aws_access_key_id     = "AKIA..."
aws_secret_access_key = "wJalr..."
openrouter_api_key    = "sk-or-v1-..."
hf_token              = "hf_..."        # leave "" if not needed

# Defaults — change only if needed
aws_region      = "us-east-1"
s3_bucket_name  = "viv-image-gen-presentation"
allowed_cidr    = "0.0.0.0/0"
public_key_path = "~/.ssh/id_rsa.pub"
```

> Terraform injects these values directly into the EC2 `.env` files at boot — you never manually SSH to configure anything.

### Minimum IAM policy for `AWS_ACCESS_KEY_ID`

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::viv-image-gen-presentation",
      "arn:aws:s3:::viv-image-gen-presentation/*"
    ]
  }]
}
```

---

## 🚀 Deployment

### Prerequisites (run once on your local machine)

```bash
# Terraform
sudo apt-get install -y gnupg software-properties-common
wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor \
  | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] \
  https://apt.releases.hashicorp.com $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt-get update && sudo apt-get install terraform

# AWS CLI
pip install awscli && aws configure   # enter keys + us-east-1

# SSH key
ls ~/.ssh/id_rsa.pub || ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
```

### Step 1 — Fill in your secrets

```bash
cd Image_generation/terraform/
nano terraform.tfvars
```

### Step 2 — Create S3 bucket

```bash
aws s3 mb s3://viv-image-gen-presentation --region us-east-1
```

### Step 3 — Deploy

```bash
cd Image_generation/terraform/
terraform init
terraform plan
terraform apply -auto-approve
```

**Output after apply:**

```
instance_public_ip   = "54.210.x.x"
instance_id          = "i-0abc123def456"
ssh_command          = "ssh -i ~/.ssh/id_rsa ubuntu@54.210.x.x"
gpu_service_url      = "http://54.210.x.x:8000"
agent_service_url    = "http://54.210.x.x:8001"
watch_bootstrap_logs = "ssh -i ~/.ssh/id_rsa ubuntu@54.210.x.x 'tail -f /var/log/sdxl-bootstrap.log'"
```

### Step 4 — Watch bootstrap (~10–15 min)

```bash
ssh -i ~/.ssh/id_rsa ubuntu@<PUBLIC_IP> 'tail -f /var/log/sdxl-bootstrap.log'
```

| Time | What's happening |
|------|-----------------|
| 0–2 min | apt update, NVIDIA driver check, Docker, NVIDIA Container Toolkit |
| 2–4 min | EBS mount + format, all project files written with secrets injected |
| 4–12 min | `download_weights.py` — ~15 GB from HuggingFace |
| 12–14 min | `docker build` both images |
| 14–17 min | Containers start, pipeline loads into VRAM, `torch.compile` warmup |
| ~17 min | Smoke test fires — service is live |

### Step 5 — Verify

```bash
# GPU check
ssh -i ~/.ssh/id_rsa ubuntu@<PUBLIC_IP> 'nvidia-smi'

# Both containers running
ssh -i ~/.ssh/id_rsa ubuntu@<PUBLIC_IP> 'docker ps'
```

Expected `docker ps`:

```
CONTAINER ID   IMAGE                PORTS                    NAMES
abc123def456   sdxl-s3-service:v1   0.0.0.0:8000->8000/tcp   sdxl_service
def456abc789   sdxl-agent:v1        0.0.0.0:8001->8001/tcp   sdxl_agent
```

---

## 🧪 API Testing

Replace `<PUBLIC_IP>` with the IP from `terraform output instance_public_ip`.

**Swagger UI:** `http://<PUBLIC_IP>:8000/docs` · `http://<PUBLIC_IP>:8001/docs`

---

### Tier 1 — Agent Pipeline (Port 8001)

**`single`** — 1 prompt → 1 image

```bash
curl -X POST "http://<PUBLIC_IP>:8001/agent/process" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Electrolysis of water experiment setup", "mode": "single", "steps": 4}'
```

**`batch`** — 1 prompt → N seed variations

```bash
curl -X POST "http://<PUBLIC_IP>:8001/agent/process" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Human heart anatomical diagram, minimalist line art", "mode": "batch", "num_images": 3, "steps": 4}'
```

**`multi-prompt`** — N perspectives → N images

```bash
curl -X POST "http://<PUBLIC_IP>:8001/agent/process" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Emperor Ashoka after Kalinga war", "mode": "multi-prompt", "num_images": 3, "steps": 4}'
```

---

### Tier 2 — Direct GPU Service (Port 8000)

```bash
# Single image
curl -X POST "http://<PUBLIC_IP>:8000/generate/single" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Chhatrapati Shivaji Maharaj in royal armor, Raigad Fort, cinematic sunset, 8k", "steps": 4}'

# Batch — 6 seed variations
curl -X POST "http://<PUBLIC_IP>:8000/generate/batch" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Chhatrapati Shivaji Maharaj in royal armor, Raigad Fort, cinematic sunset, 8k", "num_images": 6, "steps": 4}'

# Multi-style — 1 image per prompt
curl -X POST "http://<PUBLIC_IP>:8000/generate/multi-prompt" \
     -H "Content-Type: application/json" \
     -d '{
       "prompts": [
         "Chhatrapati Shivaji Maharaj on horseback, realistic oil painting style",
         "Chhatrapati Shivaji Maharaj portrait, anime style, dramatic lighting",
         "Chhatrapati Shivaji Maharaj, 3D Pixar animated character style",
         "Chhatrapati Shivaji Maharaj, traditional Indian watercolor painting style",
         "Chhatrapati Shivaji Maharaj in futuristic neon cyberpunk armor, 8k octane render"
       ],
       "steps": 4
     }'
```

---

### Response Payload (all endpoints)

```json
{
  "request_id": "20261109-045412-a1b2c3d4",
  "original_prompt": "Electrolysis of water experiment setup",
  "mode": "single",
  "enhanced_prompts_used": [
    "Scientific laboratory photo of a glass Hoffmann voltameter filled with acidified water. Platinum electrodes producing oxygen and hydrogen bubbles at anode and cathode. Clear volumetric marks, lab bench background, soft ring-light illumination, crisp reflections, textbook 3D render quality."
  ],
  "gpu_service_response": {
    "request_id": "20261109-045412-a1b2c3d4",
    "count": 1,
    "images": [
      {
        "request_id": "20261109-045412-a1b2c3d4",
        "prompt": "Scientific laboratory photo of a glass Hoffmann voltameter...",
        "s3_key": "requests/20261109-045412-a1b2c3d4/image_1.png",
        "seed_used": 3923427293,
        "url": "https://viv-image-gen-presentation.s3.amazonaws.com/requests/.../image_1.png?AWSAccessKeyId=...&Expires=1794182748"
      }
    ]
  }
}
```

---

## ⚙️ Hardware & Optimization Stack

| Optimization | Implementation | Effect |
|---|---|---|
| FP16 precision | `torch_dtype=torch.float16` | ~50% VRAM vs FP32 |
| 4-step distillation | SDXL-Lightning LoRA + `pipe.fuse_lora()` | 20–50 steps → 4 steps |
| PyTorch 2.x SDPA | native (torch 2.1.2) | replaces xformers |
| `torch.compile` | `mode="reduce-overhead"` on UNet | ~2s/image after warmup |
| Attention slicing | `pipe.enable_attention_slicing()` | prevents VRAM OOM on batches |
| VAE slicing | `pipe.enable_vae_slicing()` | prevents VRAM OOM on decode |
| In-memory streaming | `io.BytesIO` | zero disk I/O, GPU → S3 direct |
| NVMe weight cache | `/mnt/fast-scratch/sdxl-data/hf_cache` | no re-download on cold boot |
| FP16 VAE fix | `madebyollin/sdxl-vae-fp16-fix` | prevents black images on T4 |

---

## 📋 Model Reference

| Model | Source | Size | Purpose |
|---|---|---|---|
| `stable-diffusion-xl-base-1.0` | stabilityai | ~6.9 GB | Base UNet + dual CLIP encoders |
| `sdxl_lightning_4step_lora.safetensors` | ByteDance | ~394 MB | 4-step distillation LoRA |
| `sdxl-vae-fp16-fix` | madebyollin | ~335 MB | FP16-safe VAE decoder |

**Total: ~7.6 GB** cached at `/mnt/fast-scratch/sdxl-data/hf_cache`

---

## 🔧 Operational Commands

```bash
# Live logs
ssh -i ~/.ssh/id_rsa ubuntu@<IP> 'docker logs -f sdxl_service'
ssh -i ~/.ssh/id_rsa ubuntu@<IP> 'docker logs -f sdxl_agent'

# VRAM usage
ssh -i ~/.ssh/id_rsa ubuntu@<IP> 'nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 2'

# Disk usage
ssh -i ~/.ssh/id_rsa ubuntu@<IP> 'df -h /mnt/fast-scratch && du -sh /mnt/fast-scratch/sdxl-data/hf_cache'

# Restart GPU container
ssh -i ~/.ssh/id_rsa ubuntu@<IP> 'docker restart sdxl_service'

# Rebuild GPU image after code change
ssh -i ~/.ssh/id_rsa ubuntu@<IP> '
  cd ~/sdxl-tensorrt-service
  docker build -t sdxl-s3-service:v1 .
  docker stop sdxl_service && docker rm sdxl_service
  docker run -d --name sdxl_service --restart always --gpus all \
    -p 8000:8000 -v /mnt/fast-scratch/sdxl-data:/data \
    --env-file .env sdxl-s3-service:v1
'

# Rebuild agent image after code change
ssh -i ~/.ssh/id_rsa ubuntu@<IP> '
  cd ~/sdxl-tensorrt-service/prompt_enhancer
  docker build -t sdxl-agent:v1 .
  docker stop sdxl_agent && docker rm sdxl_agent
  docker run -d --name sdxl_agent --restart always \
    -p 8001:8001 --env-file .env sdxl-agent:v1
'
```

---

## 🗑️ Teardown

```bash
cd terraform/
terraform destroy -auto-approve
```

> S3 bucket is **not** destroyed by Terraform. Delete manually if needed:
> ```bash
> aws s3 rb s3://viv-image-gen-presentation --force
> ```

---

## 💰 Cost Estimate

| Resource | Type | Est. Cost |
|---|---|---|
| EC2 `g4dn.xlarge` | On-Demand us-east-1 | ~$0.526/hr |
| EBS root 80 GB gp3 | Storage | ~$6.40/mo |
| EBS scratch 100 GB gp3 | Storage | ~$8.00/mo |
| S3 storage + requests | ~50 KB/image | ~$0.023/GB |

> Use [AWS Pricing Calculator](https://calculator.aws) for precise estimates.

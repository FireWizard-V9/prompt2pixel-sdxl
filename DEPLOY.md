# Deployment Commands

## Prerequisites (run once on your local machine)

```bash
# Install Terraform
sudo apt-get install -y terraform        # Ubuntu
brew install terraform                   # macOS

# Install AWS CLI
pip install awscli
aws configure                            # enter your keys + us-east-1 region

# Ensure your SSH key exists
ls ~/.ssh/id_rsa.pub || ssh-keygen -t rsa -b 4096
```

---

## Step 1 — Fill in secrets

Edit `terraform/terraform.tfvars`:
```
aws_access_key_id     = "AKIA..."
aws_secret_access_key = "..."
openrouter_api_key    = "sk-or-v1-..."
hf_token              = "hf_..."        # leave empty string "" if not needed
```

---

## Step 2 — Create S3 bucket (if it doesn't exist yet)

```bash
aws s3 mb s3://viv-image-gen-presentation --region us-east-1
```

---

## Step 3 — Deploy infrastructure

```bash
cd terraform/

terraform init
terraform plan
terraform apply -auto-approve
```

Terraform will output:
```
instance_public_ip   = "x.x.x.x"
ssh_command          = "ssh -i ~/.ssh/id_rsa ubuntu@x.x.x.x"
gpu_service_url      = "http://x.x.x.x:8000"
agent_service_url    = "http://x.x.x.x:8001"
watch_bootstrap_logs = "ssh -i ~/.ssh/id_rsa ubuntu@x.x.x.x 'tail -f /var/log/sdxl-bootstrap.log'"
```

---

## Step 4 — Watch bootstrap progress

The user_data script runs automatically. It takes ~10-15 minutes total:
- ~2 min: NVIDIA driver + Docker + container toolkit install
- ~8 min: Model weight downloads (~15 GB)
- ~3 min: Docker image builds + container startup + GPU warmup

```bash
# Watch live bootstrap log
ssh -i ~/.ssh/id_rsa ubuntu@<PUBLIC_IP> 'tail -f /var/log/sdxl-bootstrap.log'

# Check container status
ssh -i ~/.ssh/id_rsa ubuntu@<PUBLIC_IP> 'docker ps'

# Watch GPU inference container logs (pipeline loading + warmup)
ssh -i ~/.ssh/id_rsa ubuntu@<PUBLIC_IP> 'docker logs -f sdxl_service'

# Watch agent container logs
ssh -i ~/.ssh/id_rsa ubuntu@<PUBLIC_IP> 'docker logs -f sdxl_agent'
```

---

## Step 5 — Verify GPU is working

```bash
ssh -i ~/.ssh/id_rsa ubuntu@<PUBLIC_IP> 'nvidia-smi'
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI ...   Driver Version: 535...   CUDA Version: 12.2               |
| GPU  0  Tesla T4          ...  16160MiB |
```

---

## Step 6 — Test the APIs

Replace `<PUBLIC_IP>` with the Terraform output IP.

### Test GPU service directly (Port 8000)

```bash
# Single image
curl -X POST "http://<PUBLIC_IP>:8000/generate/single" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "a red apple on a white table, studio lighting, 8k", "steps": 4}'

# Batch variations (3 seeds)
curl -X POST "http://<PUBLIC_IP>:8000/generate/batch" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Chhatrapati Shivaji Maharaj in royal armor, cinematic sunset", "num_images": 3, "steps": 4}'

# Multi-style
curl -X POST "http://<PUBLIC_IP>:8000/generate/multi-prompt" \
     -H "Content-Type: application/json" \
     -d '{"prompts": ["oil painting style warrior", "anime style warrior", "cyberpunk style warrior"], "steps": 4}'
```

### Test full pipeline via Agent (Port 8001)

```bash
# Single image via agent
curl -X POST "http://<PUBLIC_IP>:8001/agent/process" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Electrolysis of water experiment setup", "mode": "single", "steps": 4}'

# Batch via agent
curl -X POST "http://<PUBLIC_IP>:8001/agent/process" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Human heart anatomical diagram, minimalist line art", "mode": "batch", "num_images": 3, "steps": 4}'

# Multi-prompt via agent
curl -X POST "http://<PUBLIC_IP>:8001/agent/process" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Emperor Ashoka after Kalinga war", "mode": "multi-prompt", "num_images": 3, "steps": 4}'
```

### Swagger UI (browser)
- GPU service: `http://<PUBLIC_IP>:8000/docs`
- Agent service: `http://<PUBLIC_IP>:8001/docs`

---

## Useful operational commands (on the server)

```bash
# Check VRAM usage during inference
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1

# Restart GPU container (e.g. after code change)
docker restart sdxl_service

# Rebuild GPU image after code change
cd ~/sdxl-tensorrt-service
docker build -t sdxl-s3-service:v1 .
docker stop sdxl_service && docker rm sdxl_service
docker run -d --name sdxl_service --restart always --gpus all \
    -p 8000:8000 \
    -v /mnt/fast-scratch/sdxl-data:/data \
    --env-file .env \
    sdxl-s3-service:v1

# Rebuild agent image after code change
cd ~/sdxl-tensorrt-service/prompt_enhancer
docker build -t sdxl-agent:v1 .
docker stop sdxl_agent && docker rm sdxl_agent
docker run -d --name sdxl_agent --restart always \
    -p 8001:8001 \
    --env-file .env \
    sdxl-agent:v1

# Check disk usage (model weights)
df -h /mnt/fast-scratch
du -sh /mnt/fast-scratch/sdxl-data/hf_cache
```

---

## Teardown

```bash
cd terraform/
terraform destroy -auto-approve
```

> Note: The S3 bucket and its images are NOT destroyed by Terraform. Delete manually if needed:
> `aws s3 rb s3://viv-image-gen-presentation --force`

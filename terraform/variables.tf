variable "project_name" {
  description = "Prefix for all resource names"
  type        = string
  default     = "sdxl-service"
}

variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

# Deep Learning AMI GPU PyTorch 2.1 (Ubuntu 20.04) — us-east-1
# Find latest: aws ec2 describe-images --owners amazon \
#   --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu*20.04*"
variable "ami_id" {
  description = "Deep Learning AMI with NVIDIA drivers + Docker pre-installed"
  type        = string
  default     = "ami-0b0dcb5067f052a63"
}

variable "public_key_path" {
  description = "Path to your local SSH public key"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "allowed_cidr" {
  description = "CIDR allowed to reach ports 22/8000/8001. Use your IP or 0.0.0.0/0 for open."
  type        = string
  default     = "0.0.0.0/0"
}

# ── Secrets (set via terraform.tfvars or env vars) ────────────────────────────

variable "aws_access_key_id" {
  description = "AWS IAM Access Key ID for S3 uploads from the container"
  type        = string
  sensitive   = true
}

variable "aws_secret_access_key" {
  description = "AWS IAM Secret Access Key for S3 uploads from the container"
  type        = string
  sensitive   = true
}

variable "openrouter_api_key" {
  description = "OpenRouter API key for GPT-4o-Mini prompt enhancement"
  type        = string
  sensitive   = true
}

variable "hf_token" {
  description = "HuggingFace API token (optional, for gated model access)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "s3_bucket_name" {
  description = "S3 bucket for storing generated images"
  type        = string
  default     = "viv-image-gen-presentation"
}

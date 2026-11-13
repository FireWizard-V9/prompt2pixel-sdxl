output "instance_public_ip" {
  description = "Public IP of the GPU server"
  value       = aws_instance.sdxl_gpu.public_ip
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.sdxl_gpu.id
}

output "ssh_command" {
  description = "SSH into the GPU server"
  value       = "ssh -i ~/.ssh/id_rsa ubuntu@${aws_instance.sdxl_gpu.public_ip}"
}

output "gpu_service_url" {
  description = "SDXL GPU inference endpoint"
  value       = "http://${aws_instance.sdxl_gpu.public_ip}:8000"
}

output "agent_service_url" {
  description = "Prompt enhancer agent endpoint"
  value       = "http://${aws_instance.sdxl_gpu.public_ip}:8001"
}

output "watch_bootstrap_logs" {
  description = "Command to tail the bootstrap log on the server"
  value       = "ssh -i ~/.ssh/id_rsa ubuntu@${aws_instance.sdxl_gpu.public_ip} 'tail -f /var/log/sdxl-bootstrap.log'"
}

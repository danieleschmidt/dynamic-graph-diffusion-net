# Terraform Variables for DGDN Infrastructure

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "dgdn-production"
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

# Networking
variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "public_subnet_cidrs" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnet_cidrs" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]
}

# EKS Configuration
variable "cluster_endpoint_public_access_cidrs" {
  description = "List of CIDR blocks that can access the cluster endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "worker_instance_types" {
  description = "EC2 instance types for worker nodes"
  type        = list(string)
  default     = ["m6i.xlarge", "m6i.2xlarge"]
}

variable "worker_min_size" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 2
}

variable "worker_max_size" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 10
}

variable "worker_desired_size" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 3
}

variable "gpu_worker_instance_types" {
  description = "EC2 instance types for GPU worker nodes"
  type        = list(string)
  default     = ["p3.2xlarge", "p3.8xlarge"]
}

variable "gpu_worker_min_size" {
  description = "Minimum number of GPU worker nodes"
  type        = number
  default     = 0
}

variable "gpu_worker_max_size" {
  description = "Maximum number of GPU worker nodes"
  type        = number
  default     = 5
}

variable "gpu_worker_desired_size" {
  description = "Desired number of GPU worker nodes"
  type        = number
  default     = 1
}

# Database Configuration
variable "postgres_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "15.4"
}

variable "postgres_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.large"
}

variable "postgres_allocated_storage" {
  description = "Initial storage allocation for PostgreSQL"
  type        = number
  default     = 100
}

variable "postgres_max_allocated_storage" {
  description = "Maximum storage allocation for PostgreSQL"
  type        = number
  default     = 1000
}

variable "postgres_backup_retention" {
  description = "PostgreSQL backup retention period in days"
  type        = number
  default     = 7
}

variable "postgres_password" {
  description = "PostgreSQL master password"
  type        = string
  sensitive   = true
}

# Redis Configuration
variable "redis_version" {
  description = "Redis version"
  type        = string
  default     = "7.0"
}

variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.r7g.large"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes"
  type        = number
  default     = 2
}

# IAM Configuration
variable "eks_admin_users" {
  description = "List of IAM users with admin access to EKS"
  type = list(object({
    userarn  = string
    username = string
    groups   = list(string)
  }))
  default = []
}

variable "eks_admin_roles" {
  description = "List of IAM roles with admin access to EKS"
  type = list(object({
    rolearn  = string
    username = string
    groups   = list(string)
  }))
  default = []
}
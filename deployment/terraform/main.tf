# DGDN Infrastructure as Code - Production Deployment
terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  backend "s3" {
    bucket  = "dgdn-terraform-state"
    key     = "production/terraform.tfstate"
    region  = "us-east-1"
    encrypt = true
    
    dynamodb_table = "dgdn-terraform-locks"
  }
}

# Provider configurations
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "DGDN"
      Environment = var.environment
      Owner       = "DGDN-Team"
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC Configuration
resource "aws_vpc" "dgdn_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name                                    = "${var.cluster_name}-vpc"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "dgdn_igw" {
  vpc_id = aws_vpc.dgdn_vpc.id
  
  tags = {
    Name = "${var.cluster_name}-igw"
  }
}

# Public Subnets
resource "aws_subnet" "public" {
  count = length(var.availability_zones)
  
  vpc_id                  = aws_vpc.dgdn_vpc.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true
  
  tags = {
    Name                                    = "${var.cluster_name}-public-${count.index + 1}"
    Type                                    = "Public"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                = "1"
  }
}

# Private Subnets
resource "aws_subnet" "private" {
  count = length(var.availability_zones)
  
  vpc_id            = aws_vpc.dgdn_vpc.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]
  
  tags = {
    Name                                    = "${var.cluster_name}-private-${count.index + 1}"
    Type                                    = "Private"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"       = "1"
  }
}

# NAT Gateways
resource "aws_eip" "nat" {
  count = length(var.availability_zones)
  
  domain = "vpc"
  
  tags = {
    Name = "${var.cluster_name}-nat-eip-${count.index + 1}"
  }
  
  depends_on = [aws_internet_gateway.dgdn_igw]
}

resource "aws_nat_gateway" "dgdn_nat" {
  count = length(var.availability_zones)
  
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id
  
  tags = {
    Name = "${var.cluster_name}-nat-${count.index + 1}"
  }
  
  depends_on = [aws_internet_gateway.dgdn_igw]
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.dgdn_vpc.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.dgdn_igw.id
  }
  
  tags = {
    Name = "${var.cluster_name}-public-rt"
  }
}

resource "aws_route_table" "private" {
  count = length(var.availability_zones)
  
  vpc_id = aws_vpc.dgdn_vpc.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.dgdn_nat[count.index].id
  }
  
  tags = {
    Name = "${var.cluster_name}-private-rt-${count.index + 1}"
  }
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)
  
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = length(aws_subnet.private)
  
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# Security Groups
resource "aws_security_group" "eks_cluster" {
  name        = "${var.cluster_name}-cluster-sg"
  description = "Security group for EKS cluster"
  vpc_id      = aws_vpc.dgdn_vpc.id
  
  ingress {
    from_port = 443
    to_port   = 443
    protocol  = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.cluster_name}-cluster-sg"
  }
}

resource "aws_security_group" "eks_nodes" {
  name        = "${var.cluster_name}-nodes-sg"
  description = "Security group for EKS worker nodes"
  vpc_id      = aws_vpc.dgdn_vpc.id
  
  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
  }
  
  ingress {
    from_port       = 1025
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
  }
  
  ingress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.cluster_name}-nodes-sg"
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = var.cluster_name
  cluster_version = var.kubernetes_version
  
  vpc_id          = aws_vpc.dgdn_vpc.id
  subnet_ids      = aws_subnet.private[*].id
  control_plane_subnet_ids = aws_subnet.public[*].id
  
  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access_cidrs = var.cluster_endpoint_public_access_cidrs
  
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
  
  eks_managed_node_groups = {
    dgdn_workers = {
      name = "dgdn-workers"
      
      instance_types = var.worker_instance_types
      capacity_type  = "ON_DEMAND"
      
      min_size     = var.worker_min_size
      max_size     = var.worker_max_size
      desired_size = var.worker_desired_size
      
      disk_size = 100
      
      labels = {
        role = "worker"
        workload = "dgdn"
      }
      
      taints = []
      
      vpc_security_group_ids = [aws_security_group.eks_nodes.id]
    }
    
    dgdn_gpu_workers = {
      name = "dgdn-gpu-workers"
      
      instance_types = var.gpu_worker_instance_types
      capacity_type  = "ON_DEMAND"
      
      min_size     = var.gpu_worker_min_size
      max_size     = var.gpu_worker_max_size
      desired_size = var.gpu_worker_desired_size
      
      disk_size = 200
      
      labels = {
        role = "gpu-worker"
        workload = "dgdn"
        "nvidia.com/gpu" = "true"
      }
      
      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
      
      vpc_security_group_ids = [aws_security_group.eks_nodes.id]
    }
  }
  
  fargate_profiles = {
    dgdn_api = {
      name = "dgdn-api"
      selectors = [
        {
          namespace = "dgdn"
          labels = {
            workload = "api"
          }
        }
      ]
      
      subnet_ids = aws_subnet.private[*].id
      
      tags = {
        Type = "API"
      }
    }
  }
  
  manage_aws_auth_configmap = true
  
  aws_auth_users = var.eks_admin_users
  aws_auth_roles = var.eks_admin_roles
  
  tags = {
    Terraform = "true"
    Environment = var.environment
  }
}

# RDS Subnet Group
resource "aws_db_subnet_group" "dgdn_db" {
  name       = "${var.cluster_name}-db-subnet-group"
  subnet_ids = aws_subnet.private[*].id
  
  tags = {
    Name = "${var.cluster_name} DB subnet group"
  }
}

# RDS Security Group
resource "aws_security_group" "rds" {
  name        = "${var.cluster_name}-rds-sg"
  description = "Security group for RDS PostgreSQL"
  vpc_id      = aws_vpc.dgdn_vpc.id
  
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
  }
  
  tags = {
    Name = "${var.cluster_name}-rds-sg"
  }
}

# RDS PostgreSQL
resource "aws_db_instance" "dgdn_postgres" {
  identifier = "${var.cluster_name}-postgres"
  
  engine         = "postgres"
  engine_version = var.postgres_version
  instance_class = var.postgres_instance_class
  
  allocated_storage     = var.postgres_allocated_storage
  max_allocated_storage = var.postgres_max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true
  
  db_name  = "dgdn"
  username = "dgdn"
  password = var.postgres_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.dgdn_db.name
  
  backup_retention_period = var.postgres_backup_retention
  backup_window          = "03:00-04:00"
  maintenance_window     = "Sun:04:00-Sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "${var.cluster_name}-postgres-final-snapshot"
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring.arn
  
  tags = {
    Name = "${var.cluster_name}-postgres"
  }
}

# ElastiCache Subnet Group
resource "aws_elasticache_subnet_group" "dgdn_redis" {
  name       = "${var.cluster_name}-redis-subnet-group"
  subnet_ids = aws_subnet.private[*].id
}

# ElastiCache Security Group
resource "aws_security_group" "redis" {
  name        = "${var.cluster_name}-redis-sg"
  description = "Security group for ElastiCache Redis"
  vpc_id      = aws_vpc.dgdn_vpc.id
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
  }
  
  tags = {
    Name = "${var.cluster_name}-redis-sg"
  }
}

# ElastiCache Redis
resource "aws_elasticache_replication_group" "dgdn_redis" {
  replication_group_id       = "${var.cluster_name}-redis"
  description                = "Redis cluster for DGDN"
  
  port               = 6379
  parameter_group_name = "default.redis7"
  
  node_type          = var.redis_node_type
  num_cache_clusters = var.redis_num_cache_nodes
  
  engine_version = var.redis_version
  
  subnet_group_name  = aws_elasticache_subnet_group.dgdn_redis.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  snapshot_retention_limit = 5
  snapshot_window         = "03:00-05:00"
  
  tags = {
    Name = "${var.cluster_name}-redis"
  }
}

# S3 Bucket for Model Storage
resource "aws_s3_bucket" "dgdn_models" {
  bucket = "${var.cluster_name}-models-${random_string.bucket_suffix.result}"
  
  tags = {
    Name        = "DGDN Models"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "dgdn_models" {
  bucket = aws_s3_bucket.dgdn_models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "dgdn_models" {
  bucket = aws_s3_bucket.dgdn_models.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "dgdn_models" {
  bucket = aws_s3_bucket.dgdn_models.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# IAM Roles
resource "aws_iam_role" "rds_monitoring" {
  name = "${var.cluster_name}-rds-monitoring"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# Random string for unique naming
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# Outputs
output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "postgres_endpoint" {
  description = "PostgreSQL endpoint"
  value       = aws_db_instance.dgdn_postgres.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.dgdn_redis.configuration_endpoint_address
  sensitive   = true
}

output "s3_bucket_name" {
  description = "S3 bucket for model storage"
  value       = aws_s3_bucket.dgdn_models.id
}
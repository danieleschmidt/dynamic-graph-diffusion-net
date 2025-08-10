#!/bin/bash

# DGDN Production Deployment Script
# Automated deployment for DGDN research platform

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOYMENT_ENV="${1:-production}"
NAMESPACE="${DGDN_NAMESPACE:-dgdn}"
VERSION="${DGDN_VERSION:-1.0.0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_warning "helm is not installed, some features may not work"
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_info "Building DGDN Docker image..."
    
    cd "$PROJECT_ROOT"
    
    # Build production image
    docker build \
        --target production \
        --build-arg BUILD_ENV="$DEPLOYMENT_ENV" \
        --build-arg DGDN_VERSION="$VERSION" \
        -t "dgdn:$VERSION" \
        -t "dgdn:latest" \
        -f deployment/docker/Dockerfile \
        .
    
    log_success "Docker image built successfully"
}

# Push image to registry
push_image() {
    local registry="${DGDN_REGISTRY:-localhost:5000}"
    
    log_info "Pushing image to registry: $registry"
    
    # Tag for registry
    docker tag "dgdn:$VERSION" "$registry/dgdn:$VERSION"
    docker tag "dgdn:latest" "$registry/dgdn:latest"
    
    # Push to registry
    docker push "$registry/dgdn:$VERSION"
    docker push "$registry/dgdn:latest"
    
    log_success "Image pushed to registry"
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Label namespace for monitoring
    kubectl label namespace "$NAMESPACE" name="$NAMESPACE" --overwrite
    
    log_success "Namespace created/updated"
}

# Deploy secrets
deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Check if secrets exist
    if kubectl get secret dgdn-secrets -n "$NAMESPACE" &> /dev/null; then
        log_warning "Secrets already exist, skipping creation"
        return
    fi
    
    # Generate random passwords if not provided
    POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}"
    GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-$(openssl rand -base64 32)}"
    DGDN_SECRET_KEY="${DGDN_SECRET_KEY:-$(openssl rand -base64 64)}"
    
    # Create secret
    kubectl create secret generic dgdn-secrets \
        --from-literal=postgres-url="postgresql://dgdn:$POSTGRES_PASSWORD@postgres:5432/dgdn" \
        --from-literal=redis-url="redis://redis:6379" \
        --from-literal=secret-key="$DGDN_SECRET_KEY" \
        --from-literal=postgres-password="$POSTGRES_PASSWORD" \
        --from-literal=grafana-password="$GRAFANA_PASSWORD" \
        -n "$NAMESPACE"
    
    log_success "Secrets deployed"
    
    # Save passwords for reference
    cat > "$SCRIPT_DIR/passwords.txt" << EOF
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
GRAFANA_PASSWORD=$GRAFANA_PASSWORD
DGDN_SECRET_KEY=$DGDN_SECRET_KEY
EOF
    
    log_warning "Passwords saved to $SCRIPT_DIR/passwords.txt - please store securely!"
}

# Deploy ConfigMaps
deploy_configmaps() {
    log_info "Deploying ConfigMaps..."
    
    # Create ConfigMap from configs directory
    kubectl create configmap dgdn-config \
        --from-file="$PROJECT_ROOT/deployment/configs/" \
        -n "$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "ConfigMaps deployed"
}

# Deploy PersistentVolumes
deploy_storage() {
    log_info "Deploying storage..."
    
    # Apply storage configurations
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/storage.yaml" -n "$NAMESPACE"
    
    log_success "Storage deployed"
}

# Deploy DGDN application
deploy_application() {
    log_info "Deploying DGDN application..."
    
    # Update image in deployment
    sed -i.bak "s|image: dgdn:.*|image: dgdn:$VERSION|g" \
        "$PROJECT_ROOT/deployment/kubernetes/dgdn-deployment.yaml"
    
    # Apply deployments
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/dgdn-deployment.yaml" -n "$NAMESPACE"
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/dgdn-service.yaml" -n "$NAMESPACE"
    
    log_success "Application deployed"
}

# Deploy monitoring
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Deploy Prometheus and Grafana
    if [ -f "$PROJECT_ROOT/deployment/kubernetes/monitoring.yaml" ]; then
        kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/monitoring.yaml" -n "$NAMESPACE"
    fi
    
    log_success "Monitoring deployed"
}

# Wait for deployment to be ready
wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    # Wait for API deployment
    kubectl rollout status deployment/dgdn-api -n "$NAMESPACE" --timeout=600s
    
    # Wait for worker deployment
    kubectl rollout status deployment/dgdn-worker -n "$NAMESPACE" --timeout=600s
    
    log_success "Deployment is ready"
}

# Run health checks
health_check() {
    log_info "Running health checks..."
    
    # Get service endpoint
    local api_service=$(kubectl get service dgdn-api-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    # Port forward for health check
    kubectl port-forward service/dgdn-api-service 8080:80 -n "$NAMESPACE" &
    local port_forward_pid=$!
    
    sleep 5
    
    # Health check
    if curl -f http://localhost:8080/health/ready > /dev/null 2>&1; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        kill $port_forward_pid 2>/dev/null || true
        exit 1
    fi
    
    # Cleanup port forward
    kill $port_forward_pid 2>/dev/null || true
    
    log_success "All health checks passed"
}

# Display deployment information
display_info() {
    log_info "Deployment Information:"
    echo ""
    
    # Get service information
    kubectl get services -n "$NAMESPACE" -o wide
    echo ""
    
    # Get pod information
    kubectl get pods -n "$NAMESPACE" -o wide
    echo ""
    
    # Get ingress information
    if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
        echo "Ingress:"
        kubectl get ingress -n "$NAMESPACE" -o wide
        echo ""
    fi
    
    log_success "DGDN deployed successfully to $DEPLOYMENT_ENV environment!"
    
    if [ -f "$SCRIPT_DIR/passwords.txt" ]; then
        log_warning "Don't forget to securely store the passwords in $SCRIPT_DIR/passwords.txt"
    fi
}

# Rollback deployment
rollback() {
    local revision="${1:-}"
    
    log_info "Rolling back deployment..."
    
    if [ -n "$revision" ]; then
        kubectl rollout undo deployment/dgdn-api --to-revision="$revision" -n "$NAMESPACE"
        kubectl rollout undo deployment/dgdn-worker --to-revision="$revision" -n "$NAMESPACE"
    else
        kubectl rollout undo deployment/dgdn-api -n "$NAMESPACE"
        kubectl rollout undo deployment/dgdn-worker -n "$NAMESPACE"
    fi
    
    wait_for_deployment
    
    log_success "Rollback completed"
}

# Clean up deployment
cleanup() {
    log_info "Cleaning up deployment..."
    
    kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    case "${1:-deploy}" in
        "build")
            check_prerequisites
            build_image
            ;;
        "push")
            check_prerequisites
            push_image
            ;;
        "deploy")
            check_prerequisites
            build_image
            if [ "${DGDN_REGISTRY:-}" != "" ]; then
                push_image
            fi
            create_namespace
            deploy_secrets
            deploy_configmaps
            deploy_storage
            deploy_application
            deploy_monitoring
            wait_for_deployment
            health_check
            display_info
            ;;
        "upgrade")
            check_prerequisites
            build_image
            if [ "${DGDN_REGISTRY:-}" != "" ]; then
                push_image
            fi
            deploy_configmaps
            deploy_application
            wait_for_deployment
            health_check
            display_info
            ;;
        "rollback")
            rollback "${2:-}"
            ;;
        "cleanup")
            cleanup
            ;;
        "status")
            display_info
            ;;
        *)
            echo "Usage: $0 {build|push|deploy|upgrade|rollback|cleanup|status}"
            echo ""
            echo "Commands:"
            echo "  build     - Build Docker image"
            echo "  push      - Push image to registry"
            echo "  deploy    - Full deployment (build + deploy)"
            echo "  upgrade   - Upgrade existing deployment"
            echo "  rollback  - Rollback deployment [revision]"
            echo "  cleanup   - Remove deployment"
            echo "  status    - Show deployment status"
            echo ""
            echo "Environment variables:"
            echo "  DGDN_NAMESPACE    - Kubernetes namespace (default: dgdn)"
            echo "  DGDN_VERSION      - Image version tag (default: 1.0.0)"
            echo "  DGDN_REGISTRY     - Docker registry URL"
            echo "  POSTGRES_PASSWORD - PostgreSQL password"
            echo "  GRAFANA_PASSWORD  - Grafana admin password"
            echo "  DGDN_SECRET_KEY   - Application secret key"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
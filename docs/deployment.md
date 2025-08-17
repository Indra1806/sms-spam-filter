# 🚀 Deployment Guide

## Overview

This guide covers various deployment options for the SMS Spam Filter application, from local development to production cloud deployments. Choose the deployment method that best fits your needs and infrastructure requirements.

## Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Platforms](#cloud-platforms)
  - [Streamlit Cloud](#streamlit-cloud)
  - [Heroku](#heroku)
  - [AWS](#aws)
  - [Google Cloud Platform](#google-cloud-platform)
  - [Microsoft Azure](#microsoft-azure)
- [Production Considerations](#production-considerations)
- [Monitoring and Logging](#monitoring-and-logging)
- [Security](#security)
- [Scaling](#scaling)
- [Troubleshooting](#troubleshooting)

## Local Development

### Quick Start

```bash
# Clone repository
git clone https://github.com/Indra1806/sms-spam-filter.git
cd sms-spam-filter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Development Environment Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set environment variables
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export LOG_LEVEL=DEBUG

# Run with development settings
streamlit run app.py --server.runOnSave=true --server.port=8501
```

### Environment Variables

Create a `.env` file for local development:

```bash
# .env file
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
LOG_LEVEL=INFO
MODEL_DIR=models
MAX_FEATURES=5000
TEST_SIZE=0.2
RANDOM_STATE=42
```

## Docker Deployment

### Basic Docker Setup

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  sms-spam-filter:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 5
```

### Build and Run

```bash
# Build image
docker build -t sms-spam-filter .

# Run container
docker run -p 8501:8501 sms-spam-filter

# Or use docker-compose
docker-compose up -d

# View logs
docker-compose logs -f
```

### Multi-stage Build (Production)

```dockerfile
# Multi-stage Dockerfile for production
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /root/.local

# Install runtime dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Copy application
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Cloud Platforms

### Streamlit Cloud

Streamlit Cloud offers the easiest deployment for Streamlit applications.

#### Steps:

1. **Push to GitHub**: Ensure your code is in a GitHub repository
2. **Connect Repository**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `Indra1806/sms-spam-filter`
   - Set main file path: `app.py`
   - Choose branch: `main`

3. **Configuration** (optional):
   Create `.streamlit/config.toml`:
   ```toml
   [server]
   maxUploadSize = 200

   [theme]
   primaryColor = "#FF6B6B"
   backgroundColor = "#FFFFFF"
   secondaryBackgroundColor = "#F0F2F6"
   textColor = "#262730"
   ```

4. **Secrets Management**:
   Add secrets in the Streamlit Cloud dashboard:
   ```toml
   # .streamlit/secrets.toml (for local development)
   LOG_LEVEL = "INFO"
   MAX_FEATURES = "5000"
   ```

#### Advantages:
- ✅ Free tier available
- ✅ Easy GitHub integration
- ✅ Automatic deployments
- ✅ Built-in SSL certificate

### Heroku

Deploy to Heroku using Git or Docker.

#### Git Deployment:

1. **Setup Files**:
   
   Create `Procfile`:
   ```
   web: sh setup.sh && streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

   Create `setup.sh`:
   ```bash
   #!/bin/bash
   mkdir -p ~/.streamlit/
   echo "\
   [general]\n\
   email = \"your-email@domain.com\"\n\
   " > ~/.streamlit/credentials.toml
   echo "\
   [server]\n\
   headless = true\n\
   enableCORS=false\n\
   port = $PORT\n\
   " > ~/.streamlit/config.toml
   ```

   Create `runtime.txt`:
   ```
   python-3.9.16
   ```

2. **Deploy**:
   ```bash
   # Install Heroku CLI
   # Login to Heroku
   heroku login

   # Create app
   heroku create your-sms-spam-filter

   # Set environment variables
   heroku config:set LOG_LEVEL=INFO
   heroku config:set MAX_FEATURES=5000

   # Deploy
   git push heroku main

   # Open app
   heroku open
   ```

#### Docker Deployment on Heroku:

```bash
# Login to Heroku container registry
heroku container:login

# Create app
heroku create your-sms-spam-filter

# Build and push container
heroku container:push web --app your-sms-spam-filter

# Release container
heroku container:release web --app your-sms-spam-filter
```

### AWS

#### AWS Elastic Container Service (ECS)

1. **Create ECR Repository**:
   ```bash
   aws ecr create-repository --repository-name sms-spam-filter
   ```

2. **Build and Push Docker Image**:
   ```bash
   # Get login token
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

   # Build image
   docker build -t sms-spam-filter .

   # Tag image
   docker tag sms-spam-filter:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/sms-spam-filter:latest

   # Push image
   docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/sms-spam-filter:latest
   ```

3. **Create ECS Task Definition** (`task-definition.json`):
   ```json
   {
     "family": "sms-spam-filter",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "512",
     "memory": "1024",
     "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "sms-spam-filter",
         "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sms-spam-filter:latest",
         "portMappings": [
           {
             "containerPort": 8501,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "LOG_LEVEL",
             "value": "INFO"
           }
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/sms-spam-filter",
             "awslogs-region": "us-east-1",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

4. **Create ECS Service**:
   ```bash
   # Create cluster
   aws ecs create-cluster --cluster-name sms-spam-filter-cluster

   # Register task definition
   aws ecs register-task-definition --cli-input-json file://task-definition.json

   # Create service
   aws ecs create-service \
     --cluster sms-spam-filter-cluster \
     --service-name sms-spam-filter-service \
     --task-definition sms-spam-filter:1 \
     --desired-count 1 \
     --launch-type FARGATE \
     --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
   ```

#### AWS App Runner

Simpler alternative to ECS:

```yaml
# apprunner.yaml
version: 1.0
runtime: docker
build:
  commands:
    build:
      - echo "No build commands"
run:
  runtime-version: latest
  command: streamlit run app.py --server.port=8080 --server.address=0.0.0.0
  network:
    port: 8080
    env: PORT
  env:
    - name: LOG_LEVEL
      value: INFO
```

### Google Cloud Platform

#### Cloud Run Deployment

1. **Setup**:
   ```bash
   # Install Google Cloud SDK
   # Authenticate
   gcloud auth login
   gcloud config set project your-project-id
   ```

2. **Build and Deploy**:
   ```bash
   # Enable required APIs
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com

   # Deploy directly from source
   gcloud run deploy sms-spam-filter \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --port 8501 \
     --memory 1Gi \
     --cpu 1 \
     --set-env-vars LOG_LEVEL=INFO,MAX_FEATURES=5000
   ```

3. **Using Cloud Build**:
   
   Create `cloudbuild.yaml`:
   ```yaml
   steps:
   - name: 'gcr.io/cloud-builders/docker'
     args: ['build', '-t', 'gcr.io/$PROJECT_ID/sms-spam-filter', '.']
   - name: 'gcr.io/cloud-builders/docker'
     args: ['push', 'gcr.io/$PROJECT_ID/sms-spam-filter']
   - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
     entrypoint: gcloud
     args:
     - 'run'
     - 'deploy'
     - 'sms-spam-filter'
     - '--image'
     - 'gcr.io/$PROJECT_ID/sms-spam-filter'
     - '--region'
     - 'us-central1'
     - '--platform'
     - 'managed'
     - '--allow-unauthenticated'
   images:
   - 'gcr.io/$PROJECT_ID/sms-spam-filter'
   ```

#### App Engine Deployment

Create `app.yaml`:
```yaml
runtime: python39

env_variables:
  LOG_LEVEL: INFO
  MAX_FEATURES: 5000

automatic_scaling:
  min_instances: 0
  max_instances: 10

resources:
  cpu: 1
  memory_gb: 1
  disk_size_gb: 10
```

Deploy:
```bash
gcloud app deploy
```

### Microsoft Azure

#### Azure Container Instances

```bash
# Create resource group
az group create --name sms-spam-filter-rg --location eastus

# Create container instance
az container create \
  --resource-group sms-spam-filter-rg \
  --name sms-spam-filter \
  --image your-registry/sms-spam-filter:latest \
  --cpu 1 \
  --memory 1 \
  --ports 8501 \
  --dns-name-label sms-spam-filter-unique \
  --environment-variables LOG_LEVEL=INFO MAX_FEATURES=5000
```

#### Azure App Service

1. **Create App Service Plan**:
   ```bash
   az appservice plan create \
     --name sms-spam-filter-plan \
     --resource-group sms-spam-filter-rg \
     --sku B1 \
     --is-linux
   ```

2. **Create Web App**:
   ```bash
   az webapp create \
     --resource-group sms-spam-filter-rg \
     --plan sms-spam-filter-plan \
     --name sms-spam-filter-app \
     --deployment-container-image-name your-registry/sms-spam-filter:latest
   ```

3. **Configure App Settings**:
   ```bash
   az webapp config appsettings set \
     --resource-group sms-spam-filter-rg \
     --name sms-spam-filter-app \
     --settings LOG_LEVEL=INFO MAX_FEATURES=5000 WEBSITES_PORT=8501
   ```

## Production Considerations

### Security Hardening

#### Environment Variables
```bash
# Production environment variables
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

#### Secure Configuration
Create `.streamlit/config.toml`:
```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
base = "light"
```

#### Docker Security
```dockerfile
# Use non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Set proper file permissions
COPY --chown=app:app . .

# Remove unnecessary packages
RUN apt-get remove -y build-essential && apt-get autoremove -y
```

### Performance Optimization

#### Resource Limits
```yaml
# docker-compose.yml
services:
  sms-spam-filter:
    build: .
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
```

#### Caching Strategy
```python
import streamlit as st
import functools

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_models():
    """Cached model loading"""
    spam_filter = SMSSpamFilter()
    spam_filter.load_models()
    return spam_filter

@functools.lru_cache(maxsize=1000)
def cached_prediction(message_hash, model_name):
    """Cache predictions for identical messages"""
    spam_filter = load_models()
    return spam_filter.predict(message_hash, model_name)
```

#### Model Optimization
```python
# Optimize model size
from sklearn.feature_extraction.text import TfidfVectorizer

# Reduce vocabulary size for production
vectorizer = TfidfVectorizer(
    max_features=3000,  # Reduced from 5000
    max_df=0.9,        # Remove very common words
    min_df=3,          # Remove very rare words
    dtype=np.float32   # Use 32-bit floats
)
```

### Health Checks

#### Application Health Check
```python
# health_check.py
import requests
import sys

def health_check(url="http://localhost:8501"):
    try:
        response = requests.get(f"{url}/_stcore/health", timeout=10)
        if response.status_code == 200:
            print("✅ Application is healthy")
            return 0
        else:
            print(f"❌ Application unhealthy: {response.status_code}")
            return 1
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(health_check())
```

#### Kubernetes Health Checks
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sms-spam-filter
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sms-spam-filter
  template:
    metadata:
      labels:
        app: sms-spam-filter
    spec:
      containers:
      - name: sms-spam-filter
        image: your-registry/sms-spam-filter:latest
        ports:
        - containerPort: 8501
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## Monitoring and Logging

### Application Logging

#### Structured Logging
```python
import json
import logging
import sys
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'line': record.lineno
        }
        return json.dumps(log_entry)

# Configure logging
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    return logger

# Usage in app.py
logger = setup_logging()

@st.cache_data
def predict_with_logging(message, model_name):
    logger.info(f"Prediction request: model={model_name}, message_length={len(message)}")
    
    try:
        result, confidence = spam_filter.predict(message, model_name)
        logger.info(f"Prediction result: {result} ({confidence:.2%})")
        return result, confidence
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return None, 0.0
```

### Metrics Collection

#### Prometheus Metrics
```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions', ['model', 'result'])
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Prediction duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')

def track_prediction(model_name, result, duration):
    PREDICTION_COUNTER.labels(model=model_name, result=result).inc()
    PREDICTION_DURATION.observe(duration)

def start_metrics_server():
    start_http_server(8000)  # Metrics available at :8000/metrics
```

#### Application Metrics
```python
# app.py modifications
import time
from collections import defaultdict

# Global metrics storage
metrics = {
    'total_predictions': 0,
    'predictions_by_model': defaultdict(int),
    'predictions_by_result': defaultdict(int),
    'average_confidence': []
}

def update_metrics(model_name, result, confidence):
    metrics['total_predictions'] += 1
    metrics['predictions_by_model'][model_name] += 1
    metrics['predictions_by_result'][result] += 1
    metrics['average_confidence'].append(confidence)

# Display metrics in sidebar
def display_metrics():
    with st.sidebar:
        st.subheader("📊 Application Metrics")
        st.metric("Total Predictions", metrics['total_predictions'])
        
        if metrics['average_confidence']:
            avg_conf = sum(metrics['average_confidence']) / len(metrics['average_confidence'])
            st.metric("Avg Confidence", f"{avg_conf:.2%}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Spam Detected", metrics['predictions_by_result']['spam'])
        with col2:
            st.metric("Ham Detected", metrics['predictions_by_result']['ham'])
```

### Error Tracking

#### Sentry Integration
```python
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

# Initialize Sentry
sentry_logging = LoggingIntegration(
    level=logging.INFO,
    event_level=logging.ERROR
)

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[sentry_logging],
    traces_sample_rate=0.1
)

# Usage
try:
    result, confidence = spam_filter.predict(message, model_name)
except Exception as e:
    sentry_sdk.capture_exception(e)
    st.error("An error occurred during prediction")
```

## Scaling

### Horizontal Scaling

#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sms-spam-filter
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sms-spam-filter
  template:
    spec:
      containers:
      - name: app
        image: sms-spam-filter:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sms-spam-filter-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sms-spam-filter
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### Load Balancer Configuration
```yaml
apiVersion: v1
kind: Service
metadata:
  name: sms-spam-filter-service
spec:
  selector:
    app: sms-spam-filter
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sms-spam-filter-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: sms-spam-filter.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: sms-spam-filter-service
            port:
              number: 80
```

### Vertical Scaling

#### Resource Optimization
```python
# config.py
import os

class ScalingConfig:
    # Scale based on environment
    ENV = os.getenv('ENV', 'development')
    
    if ENV == 'production':
        MAX_FEATURES = 5000
        MODEL_CACHE_SIZE = 1000
        BATCH_SIZE = 100
    elif ENV == 'staging':
        MAX_FEATURES = 3000
        MODEL_CACHE_SIZE = 500
        BATCH_SIZE = 50
    else:  # development
        MAX_FEATURES = 1000
        MODEL_CACHE_SIZE = 100
        BATCH_SIZE = 10
```

## Troubleshooting

### Common Issues

#### 1. Memory Issues
```bash
# Check memory usage
docker stats sms-spam-filter

# Increase memory limit
docker run -m 2g sms-spam-filter

# Monitor in Kubernetes
kubectl top pods
```

**Solution:**
```python
# Optimize memory usage
import gc

@st.cache_data(max_entries=100)  # Limit cache size
def optimized_predict(message, model_name):
    result = spam_filter.predict(message, model_name)
    gc.collect()  # Force garbage collection
    return result
```

#### 2. Model Loading Failures
```python
def robust_model_loading():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if spam_filter.load_models():
                return True
            else:
                logger.warning(f"Model loading attempt {attempt + 1} failed")
        except Exception as e:
            logger.error(f"Model loading error: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
    
    # Fallback to training
    logger.info("Falling back to model training")
    return spam_filter.train_models()
```

#### 3. Port Binding Issues
```bash
# Check if port is in use
netstat -tulpn | grep :8501

# Kill process using port
sudo fuser -k 8501/tcp

# Use different port
streamlit run app.py --server.port=8502
```

#### 4. SSL/TLS Issues
```bash
# For Streamlit Cloud
echo '[server]
enableCORS = false
enableXsrfProtection = true
' > .streamlit/config.toml

# For custom domains
# Add SSL certificate to load balancer
```

### Debugging Tools

#### Container Debugging
```bash
# Access running container
docker exec -it sms-spam-filter /bin/bash

# Check logs
docker logs sms-spam-filter --tail 100 -f

# Debug health check
docker exec sms-spam-filter curl http://localhost:8501/_stcore/health
```

#### Kubernetes Debugging
```bash
# Check pod status
kubectl get pods -l app=sms-spam-filter

# View pod logs
kubectl logs -f deployment/sms-spam-filter

# Debug pod
kubectl exec -it sms-spam-filter-pod -- /bin/bash

# Port forward for testing
kubectl port-forward service/sms-spam-filter-service 8501:80
```

### Performance Debugging

#### Application Profiling
```python
import cProfile
import pstats

def profile_prediction(message):
    pr = cProfile.Profile()
    pr.enable()
    
    result, confidence = spam_filter.predict(message)
    
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
    
    return result, confidence
```

#### Memory Profiling
```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Your function here
    pass
```

## Backup and Disaster Recovery

### Model Backup Strategy

#### Automated Backups
```bash
#!/bin/bash
# backup_models.sh

BACKUP_DIR="/backups/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$BACKUP_DIR"

# Backup model files
cp -r models/ "$BACKUP_DIR/"

# Upload to cloud storage
aws s3 sync "$BACKUP_DIR" s3://your-backup-bucket/sms-spam-filter/

# Keep only last 30 days of backups
find /backups -type d -mtime +30 -exec rm -rf {} +
```

#### Database Backup (if applicable)
```python
import sqlite3
import shutil
from datetime import datetime

def backup_database():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f"backup_db_{timestamp}.sqlite"
    
    # Create backup
    shutil.copy2('app.db', backup_file)
    
    # Upload to cloud storage
    # Implementation depends on your cloud provider
```

### Disaster Recovery Plan

1. **Automated Health Checks**: Continuous monitoring
2. **Failover Strategy**: Multiple deployment regions
3. **Data Recovery**: Regular model and data backups
4. **Rollback Procedure**: Quick deployment rollback
5. **Communication Plan**: Status page and notifications

## Conclusion

This deployment guide covers various options from simple local development to complex production deployments. Choose the approach that best fits your requirements:

- **Development**: Local setup with virtual environment
- **Quick Deploy**: Streamlit Cloud or Heroku
- **Production**: Docker + Kubernetes on AWS/GCP/Azure
- **Enterprise**: Multi-region deployment with monitoring

For additional support or questions about deployment, please refer to the [main documentation](../README.md) or open an issue on the [GitHub repository](https://github.com/Indra1806/sms-spam-filter).

## Quick Reference

### Essential Commands
```bash
# Local development
streamlit run app.py

# Docker
docker build -t sms-spam-filter .
docker run -p 8501:8501 sms-spam-filter

# Kubernetes
kubectl apply -f deployment.yaml
kubectl get pods -l app=sms-spam-filter

# Health check
curl http://localhost:8501/_stcore/health
```

### Environment Variables
```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
LOG_LEVEL=INFO
MAX_FEATURES=5000
MODEL_DIR=models
```

### Useful URLs
- Local development: `http://localhost:8501`
- Health check: `http://localhost:8501/_stcore/health`
- Metrics (if enabled): `http://localhost:8000/metrics`
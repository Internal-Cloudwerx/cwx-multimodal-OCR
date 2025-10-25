# Setup Guide

## Prerequisites

- Python 3.8+
- Google Cloud Platform account
- Google Document AI processor set up
- Authenticated with Google Cloud (no manual key files needed!)

## Multi-Agent System Overview

This project implements a multi-agent document analysis system with:
- **Vision Specialist Agent**: Handles images, photos, figures, diagrams
- **OCR Specialist Agent**: Processes tables, forms, structured data
- **Layout Specialist Agent**: Analyzes document structure and layout
- **Orchestrator Agent**: Routes questions to optimal specialist
- **Answer Validator Agent**: Provides confidence scoring
- **ADK A2A Wrapper**: Enables agent-to-agent communication

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set Up Google Cloud

### 2.1 Create a GCP Project
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select an existing one
3. Note your project ID

### 2.2 Enable Required APIs
```bash
gcloud services enable documentai.googleapis.com
gcloud services enable aiplatform.googleapis.com
```

### 2.3 Create Document AI Processor
1. Go to [Document AI Console](https://console.cloud.google.com/ai/document-ai)
2. Create a new processor (type: "Document OCR")
3. Note the Processor ID

### 2.4 Authenticate with Google Cloud (Easy Method)

**Option A: Using gcloud CLI (Recommended)**
```bash
# Install gcloud CLI if you haven't already
# https://cloud.google.com/sdk/docs/install

# Authenticate with your Google account
gcloud auth login

# Set application default credentials
gcloud auth application-default login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

## Step 3: Configure Environment

### 3.1 Create .env File
```bash
cp .env.example .env
```

### 3.2 Edit .env with Your Values
```bash
# Open .env and fill in your actual values:

GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT=your-actual-project-id
GOOGLE_CLOUD_LOCATION=us-central1
MODEL=gemini-2.5-flash

GCP_PROJECT_ID=your-actual-project-id
GCP_LOCATION=us
DOCUMENT_AI_PROCESSOR_ID=your-actual-processor-id

```

## Step 4: Test Multi-Agent System

### 4.1 Test Individual Agents
```bash
# Test Vision Specialist
python3 scripts/test_vision_specialist.py

# Test OCR Specialist
python3 scripts/test_ocr_specialist.py

# Test Layout Specialist
python3 scripts/test_layout_specialist.py

# Test Orchestrator
python3 scripts/test_orchestrator.py
```

### 4.2 Test Multi-Agent Integration
```bash
# Quick multi-agent test
python3 evaluation/multi_agent_evaluator.py --num-samples 5

# A2A communication demo
python3 scripts/demo_adk_a2a.py
```

## Step 5: Run Evaluation (Optional)

```bash
# Test on 10 samples
python3 evaluation/multi_agent_evaluator.py --num-samples 10

# Full validation set (requires higher API quotas)
python3 evaluation/multi_agent_evaluator.py --num-samples 5349
```

## Step 6: Deploy with ADK Web (Optional)

```bash
# Start interactive web UI
adk web --port 4200

# Open browser to http://localhost:4200
```

## Troubleshooting

### "403 Forbidden" Error
- Check your service account has the right permissions
- Verify APIs are enabled in GCP

### "Processor not found" Error
- Double-check your `DOCUMENT_AI_PROCESSOR_ID`
- Verify processor is in the same project/location

### "429 Rate Limit" Error
- Request quota increase: https://console.cloud.google.com/iam-admin/quotas
- Or add rate limiting in evaluation code (see evaluation/README.md)

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python3 --version` (should be 3.8+)

## Cost Estimates

- Document AI OCR: ~$0.0015 per page
- Gemini 2.5 Flash: ~$0.0001 per request
- **Total: ~$0.0016 per document**

For 100 documents: ~$0.16
For 1000 documents: ~$1.60

## Need Help?

- Check [evaluation/README.md](evaluation/README.md) for benchmark details
- Review the main [README.md](README.md) for project overview
- Open an issue on GitHub


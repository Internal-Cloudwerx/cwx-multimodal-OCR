# (WIP) cwx-multimodal-OCR

A multi-modal, multi-agentic approach to analyze documents using Gemini and DocAI OCR for one-shot benchmark/enterprise-ready uses

Next steps:
- For the weak points of this pipeline fill in with agents
- For example, "Image Specialist Agent", "Router/Orchestrator Agent", etc.
- Add ADK wrappers for web support 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements a hybrid approach combining:
- Google Document AI for OCR and structured text extraction
- Gemini 2.5 Flash Vision for visual document understanding
- Selective strategy that intelligently chooses when to use OCR vs. vision-only

## Features

### Current Implementation
- Hybrid OCR + Vision pipeline
- Selective OCR strategy based on question type
- Zero-shot learning (no training required)
- Rate-limited API calls for reliability
- Benchmark evaluation on SP-DocVQA dataset
- ADK (Agent Development Kit) integration


## Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/document_ai_agent.git
cd document_ai_agent
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root with your credentials:
```bash
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
MODEL=gemini-2.5-flash

GCP_PROJECT_ID=your-project-id
GCP_LOCATION=us
DOCUMENT_AI_PROCESSOR_ID=your-processor-id

# Authentication: Use gcloud CLI
```

### Usage

**Test on sample documents:**
```bash
python3 scripts/test_docvqa_quick.py
```

**Run benchmark evaluation:**
```bash
# Small test
python3 scripts/run_docvqa_benchmark.py --num-samples 10

# Larger evaluation
python3 scripts/run_docvqa_benchmark.py --num-samples 100
```

**Deploy interactive web interface:**
```bash
adk web --port 4200
# Open http://localhost:4200
```

## How It Works

### Pipeline Overview

```
Document + Question
        ↓
   [Question Type Analysis]
   Is it structured (table/form)?
        ↓
    ┌───┴───┐
   Yes     No
    ↓       ↓
  [OCR]  [Vision Only]
    └───┬───┘
        ↓
  [Gemini Vision]
  (Image + OCR context)
        ↓
    Answer
```

### Key Components

1. **Question Type Classifier**: Determines optimal processing strategy
2. **Document AI OCR**: Extracts text and tables from structured documents
3. **Gemini Vision**: Analyzes document images directly
4. **Answer Generator**: Produces concise, accurate answers

## Project Structure

```
cwx-multimodal-OCR/
├── agent.py                     # ADK agent definition
├── tools/
│   └── document_ocr.py         # Document AI integration
├── evaluation/
│   ├── docvqa_evaluator.py     # Benchmark evaluator
│   ├── hf_docvqa_loader.py     # Dataset loader
│   ├── anls_metric.py          # ANLS scoring
│   └── README.md               # Evaluation guide
├── scripts/
│   ├── run_docvqa_benchmark.py # Run evaluations
│   └── test_docvqa_quick.py    # Quick tests
├── data/
│   └── sample/                 # Sample documents
├── credentials/                 # GCP credentials (gitignored)
├── SETUP.md                     # Detailed setup guide
├── README.md                    # This file
└── requirements.txt             # Dependencies
```

## Benchmarking

### DocVQA Evaluation

The system can be evaluated on the SP-DocVQA dataset:

```bash
# Quick test (10 samples)
python3 scripts/run_docvqa_benchmark.py --num-samples 10

# Medium test (100 samples)
python3 scripts/run_docvqa_benchmark.py --num-samples 100

# Full validation set (5,349 samples, requires higher API quotas)
python3 scripts/run_docvqa_benchmark.py --num-samples 5349
```

Results are saved to `evaluation/results/docvqa/` including:
- Detailed per-sample results (JSON)
- Aggregated metrics (JSON)
- CSV export for analysis

See [evaluation/README.md](evaluation/README.md) for more details.

### Metrics

The evaluation uses the **ANLS (Average Normalized Levenshtein Similarity)** metric, which is the standard for DocVQA benchmarks:
- 1.0 = Perfect match
- ≥ 0.5 = Considered correct
- < 0.5 = Considered incorrect

## Configuration Options

### Model Selection

```bash
# In .env file
MODEL=gemini-2.5-flash  # Recommended for vision tasks
# or
MODEL=gemini-1.5-pro    # Alternative
```

### Rate Limiting

To avoid API rate limits during large evaluations:

```python
# In evaluation/docvqa_evaluator.py
self.min_request_interval = 2.0  # Seconds between requests
```

### Selective OCR Strategy

Configure which question types use OCR in `evaluation/docvqa_evaluator.py`:

```python
ocr_beneficial_types = {
    'table/list',      # Tables
    'figure/diagram',  # Charts
    'layout',          # Layout analysis
    'form',            # Forms
}
```

## Development

### Adding New Tools

Create tools in `tools/` directory:

```python
from google.adk.agents import agent_tool

@agent_tool(description="Your tool description")
def your_tool(param: str) -> str:
    """Tool implementation"""
    return result
```

Register in `agent.py`:

```python
from tools.your_module import your_tool

root_agent = Agent(
    name="document_processing_agent",
    model=os.getenv('MODEL', 'gemini-2.5-flash'),
    tools=[
        process_document_with_ocr,
        search_document,
        your_tool  # Add here
    ]
)
```

### Running Tests

```bash
# Quick functionality test
python3 scripts/test_docvqa_quick.py

# Small benchmark
python3 scripts/run_docvqa_benchmark.py --num-samples 10
```

## Cost Estimates

Approximate costs per document:
- Document AI OCR: ~$0.0015
- Gemini 2.5 Flash: ~$0.0001
- **Total: ~$0.0016 per document**

Scaling:
- 100 documents: ~$0.16
- 1,000 documents: ~$1.60
- 10,000 documents: ~$16.00

## Performance

The system achieves competitive performance on document question answering benchmarks using a hybrid approach:
- **Structured documents** (tables, forms): Excellent performance with OCR + Vision
- **Visual content** (photos, images): Direct vision processing
- **Mixed documents**: Intelligent strategy selection based on question type

Results are reproducible using the benchmark evaluation scripts in `evaluation/`.

---

## Troubleshooting

### Common Issues

**"403 Forbidden" Error**
- Verify you're authenticated: `gcloud auth list`
- Check APIs are enabled in GCP console
- Ensure your account has required permissions

**"Processor not found" Error**
- Double-check `DOCUMENT_AI_PROCESSOR_ID` in .env
- Ensure processor is in same project/location

**"429 Rate Limit" Error**
- Increase `min_request_interval` in evaluator
- Request quota increase via GCP console
- See [evaluation/README.md](evaluation/README.md) for details

**Import Errors**
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python3 --version` (requires 3.8+)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Dependencies

Main dependencies (see `requirements.txt` for complete list):
- `google-adk` - Agent Development Kit
- `google-cloud-documentai` - Document AI client
- `google-cloud-aiplatform` - Vertex AI
- `datasets` - HuggingFace datasets (for DocVQA)
- `python-Levenshtein` - ANLS metric calculation
- `tqdm` - Progress bars
- `pandas` - Result analysis

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Resources

- [Google Document AI Documentation](https://cloud.google.com/document-ai/docs)
- [Vertex AI Gemini Documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini)
- [SP-DocVQA Dataset](https://huggingface.co/datasets/lmms-lab/DocVQA)
- [ADK Documentation](https://github.com/google/adk)

## Acknowledgments

This project uses:
- Google Document AI for OCR and document understanding
- Google Gemini Vision for visual document analysis
- HuggingFace DocVQA dataset for evaluation
- Google ADK (Agent Development Kit) for agent orchestration

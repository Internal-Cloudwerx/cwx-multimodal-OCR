# Multi-Agent DocVQA Evaluation System

A production-ready evaluation system for benchmarking multi-agent document processing systems on the SP-DocVQA dataset using HuggingFace datasets.

## Overview

This evaluation system allows you to:
- Automatically download SP-DocVQA dataset from HuggingFace
- Evaluate multi-agent document processing systems on standard benchmarks
- Compute official ANLS (Average Normalized Levenshtein Similarity) scores
- Generate detailed reports by question type and agent routing
- Track agent-to-agent communication and confidence scores
- Resume interrupted evaluations

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `datasets` - HuggingFace datasets library
- `python-Levenshtein` - For ANLS metric
- `tqdm` - Progress bars
- `pandas` - For CSV reports
- `Pillow` - Image handling (already installed)

## Quick Start

### Step 1: Verify Setup

Run the quick test to verify everything is working:

```bash
python scripts/test_docvqa_quick.py
```

This will:
- Download a small portion of the validation set
- Test dataset loading
- Verify ANLS metric implementation
- Show sample questions

**Expected output:**
```
Quick DocVQA Setup Test
================================================================================

1. Testing dataset loading...
Loading SP-DocVQA validation split from Hugging Face...
✓ Loaded 5349 samples
✓ Dataset loaded successfully

2. Testing sample retrieval...
✓ Sample retrieved
  Question: What is the date?
  Answers: ['05/04/1999', 'May 4, 1999']
  Image type: <class 'PIL.Image.Image'>

3. Testing ANLS metric...
  ✓ 'exact' vs ['exact']: 1.000
  ✓ 'close match' vs ['close match']: 1.000
  ✓ '45.67' vs ['$45.67']: 0.833

================================================================================
✓ All tests passed!
================================================================================
```

### Step 2: Mini Test (10 samples)

Test your multi-agent system on 10 samples to verify integration:

```bash
python evaluation/multi_agent_evaluator.py --num-samples 10
```

**Cost:** ~$0.10 (10 Document AI calls + Gemini tokens)

### Step 3: Medium Test (100 samples)

Get meaningful benchmark results:

```bash
python evaluation/multi_agent_evaluator.py --num-samples 100
```

**Cost:** ~$1.00

### Step 4: Full Validation Set

Run on the complete validation set (~5,000 samples):

```bash
python evaluation/multi_agent_evaluator.py --split validation
```

**Cost:** ~$4-5 (one-time cost for full benchmark)

## Usage

### Basic Usage

```bash
# Evaluate on 50 samples
python scripts/run_docvqa_benchmark.py --num-samples 50

# Evaluate on validation set
python scripts/run_docvqa_benchmark.py --split validation

# Evaluate on training set
python scripts/run_docvqa_benchmark.py --split train
```

### Advanced Options

```bash
# Resume from interrupted run
python scripts/run_docvqa_benchmark.py --num-samples 1000
# (Automatically resumes if progress file exists)

# Start fresh (don't resume)
python scripts/run_docvqa_benchmark.py --num-samples 1000 --no-resume

# Start from specific index
python scripts/run_docvqa_benchmark.py --start-idx 500 --num-samples 500

# Custom output directory
python scripts/run_docvqa_benchmark.py --output-dir my_results/

# Custom cache directory for dataset
python scripts/run_docvqa_benchmark.py --cache-dir /path/to/cache
```

## Output Files

Each evaluation run generates three files:

### 1. Detailed Results JSON (`detailed_results_TIMESTAMP.json`)

Contains all predictions, ground truth, and metadata:

```json
{
  "metadata": {
    "dataset": "SP-DocVQA-validation",
    "timestamp": "20251022_143045",
    "agent": "document_processing_agent"
  },
  "metrics": {
    "overall_anls": 0.7234,
    "accuracy": 0.7551,
    "by_question_type": {...}
  },
  "results": [
    {
      "questionId": 123,
      "question": "What is the total amount?",
      "prediction": "$45.67",
      "ground_truth": ["$45.67", "45.67"],
      "anls_score": 1.0,
      "correct": true,
      "question_types": ["numerical"],
      "processing_time": 3.2
    }
  ]
}
```

### 2. Metrics Summary (`metrics_TIMESTAMP.json`)

Quick summary of performance:

```json
{
  "overall_anls": 0.7234,
  "accuracy": 0.7551,
  "by_question_type": {
    "extractive": {"anls": 0.78, "count": 65},
    "abstractive": {"anls": 0.61, "count": 20}
  },
  "avg_processing_time_seconds": 3.2
}
```

### 3. Results CSV (`results_TIMESTAMP.csv`)

Spreadsheet format for analysis in Excel/Sheets.

## Understanding ANLS

**ANLS (Average Normalized Levenshtein Similarity)** is the standard metric for DocVQA tasks.

### How it works:

1. Compute edit distance between prediction and ground truth
2. Normalize by max string length: `similarity = 1 - (edit_distance / max_length)`
3. Apply threshold (0.5): scores below 0.5 become 0
4. For multiple ground truths, take the maximum score

### Examples:

| Prediction | Ground Truth | ANLS Score |
|------------|-------------|------------|
| `$45.67` | `$45.67` | 1.0 (exact) |
| `45.67` | `$45.67` | 0.833 (close) |
| `SuperMart` | `Super Mart` | 0.9 (minor diff) |
| `wrong` | `correct` | 0.0 (below threshold) |

### Performance Baselines:

- **ANLS > 0.3** = Reasonable performance
- **ANLS > 0.5** = Competitive with traditional OCR + QA
- **ANLS > 0.7** = Strong performance (multimodal models)
- **ANLS > 0.8** = State-of-the-art territory

## Example Output

```
================================================================================
EVALUATION COMPLETE
================================================================================
Dataset: SP-DocVQA-validation
Total Samples: 100
Successful: 98
Failed: 2

PERFORMANCE METRICS:
  Overall ANLS: 0.7234 (72.34%)
  Accuracy (ANLS >= 0.5): 0.7551 (75.51%)

BY QUESTION TYPE:
  extractive:
    ANLS: 0.7845 (78.45%)
    Count: 65
  abstractive:
    ANLS: 0.6123 (61.23%)
    Count: 20
  arithmetic:
    ANLS: 0.6890 (68.90%)
    Count: 13

PERFORMANCE:
  Avg Time: 3.2s
  Median Time: 2.8s
  Total Time: 313.6s
================================================================================
```

## Cost Estimation

### Per-Sample Costs:

- **Document AI OCR:** $0.0015 per page
- **Gemini 2.5 Flash:** ~$0.0001 per question (estimated)
- **Total per sample:** ~$0.0016

### Full Evaluation Costs:

| Samples | Approx Cost | Use Case |
|---------|-------------|----------|
| 10 | $0.02 | Quick test |
| 50 | $0.08 | Initial test |
| 100 | $0.16 | Development |
| 500 | $0.80 | Thorough test |
| 5,000 | $8.00 | Full benchmark |

**Note:** Costs are approximate and depend on document complexity and model usage.

## Troubleshooting

### Issue: Dataset download is slow

**Solution:** The first download can take 10-15 minutes as HuggingFace downloads images. Subsequent runs use the cache.

```bash
# Use custom cache location with more space
python scripts/run_docvqa_benchmark.py --cache-dir /path/to/large/disk
```

### Issue: Agent integration fails

**Error:** `ADK runner failed`

**Solution:** The evaluator has a fallback mode that calls your OCR tool directly. Check these:

1. Ensure `agent.py` contains `root_agent`
2. Verify ADK is installed: `pip install google-adk`
3. Check agent logs in the output

### Issue: Out of memory

**Solution:** Process in smaller batches:

```bash
# Process 100 at a time
python scripts/run_docvqa_benchmark.py --start-idx 0 --num-samples 100
python scripts/run_docvqa_benchmark.py --start-idx 100 --num-samples 100
```

### Issue: Need to resume

Evaluation automatically saves progress every 10 samples. If interrupted:

```bash
# Just run the same command - it will resume automatically
python scripts/run_docvqa_benchmark.py --num-samples 1000
```

To start fresh:

```bash
python scripts/run_docvqa_benchmark.py --num-samples 1000 --no-resume
```

## Architecture

```
evaluation/
├── hf_docvqa_loader.py      # HuggingFace dataset loader
├── anls_metric.py            # ANLS metric implementation
├── docvqa_evaluator.py       # Main evaluator with agent integration
├── data/
│   └── docvqa_hf/            # Auto-cached by HuggingFace
└── results/
    └── docvqa/               # Evaluation outputs
```

## Dataset Information

**SP-DocVQA** (Single Page Document Visual Question Answering)
- **Source:** HuggingFace `rubentito/docvqa`
- **Splits:** train, validation, test
- **Validation size:** ~5,000 questions on ~2,500 documents
- **Question types:** Extractive, abstractive, arithmetic, counting, etc.

## Citing Your Results

When presenting benchmark results:

> "Evaluated on the SP-DocVQA validation set (Mathew et al., 2021), achieving an ANLS score of X.XX% on 5,000 questions."

**Reference:**
```
Mathew, M., Karatzas, D., & Jawahar, C. V. (2021).
DocVQA: A Dataset for VQA on Document Images.
In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 2200-2209).
```

## Next Steps

1. **Run quick test:** `python scripts/test_docvqa_quick.py`
2. **Test on 10 samples:** `python scripts/run_docvqa_benchmark.py --num-samples 10`
3. **Analyze errors:** Review `detailed_results_*.json` for failure cases
4. **Improve agent:** Iterate on weak question types
5. **Full evaluation:** `python scripts/run_docvqa_benchmark.py`
6. **Report results:** Use metrics in your presentation

## Support

For issues or questions:
1. Check logs in console output
2. Review `detailed_results_*.json` for error messages
3. Ensure all dependencies are installed
4. Verify agent is working: test on a single document first


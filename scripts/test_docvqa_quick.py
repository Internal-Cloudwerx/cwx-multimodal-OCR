#!/usr/bin/env python3
"""
Quick test on a few samples to verify setup
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.hf_docvqa_loader import HFDocVQADataset
from evaluation.anls_metric import anls_score_multi
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Quick DocVQA Setup Test")
    logger.info("="*80)
    
    # Test 1: Load dataset
    logger.info("\n1. Testing dataset loading...")
    try:
        loader = HFDocVQADataset()
        dataset = loader.load("validation")
        logger.info("✓ Dataset loaded successfully")
    except Exception as e:
        logger.error(f"✗ Dataset loading failed: {e}")
        return
    
    # Test 2: Get sample
    logger.info("\n2. Testing sample retrieval...")
    try:
        sample = loader.get_sample(0)
        logger.info(f"✓ Sample retrieved")
        logger.info(f"  Question: {sample['question']}")
        logger.info(f"  Answers: {sample['answers']}")
        logger.info(f"  Image type: {type(sample['image'])}")
    except Exception as e:
        logger.error(f"✗ Sample retrieval failed: {e}")
        return
    
    # Test 3: ANLS metric
    logger.info("\n3. Testing ANLS metric...")
    try:
        test_cases = [
            ("exact", ["exact"], 1.0),
            ("close match", ["close match"], 1.0),
            ("45.67", ["$45.67"], 0.8),
        ]
        for pred, gt, expected_min in test_cases:
            score = anls_score_multi(pred, gt)
            status = "✓" if score >= expected_min else "✗"
            logger.info(f"  {status} '{pred}' vs {gt}: {score:.3f}")
    except Exception as e:
        logger.error(f"✗ ANLS metric failed: {e}")
        return
    
    logger.info("\n" + "="*80)
    logger.info("✓ All tests passed!")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("1. Test on 10 samples: python scripts/run_docvqa_benchmark.py --num-samples 10")
    logger.info("2. Full validation: python scripts/run_docvqa_benchmark.py")

if __name__ == "__main__":
    main()


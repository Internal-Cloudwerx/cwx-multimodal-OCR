#!/usr/bin/env python3
"""
Run DocVQA benchmark evaluation on your document processing agent
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.docvqa_evaluator import DocVQAEvaluator
from evaluation.docvqa_evaluator_v2 import DocVQAEvaluatorV2
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate agent on SP-DocVQA benchmark"
    )
    parser.add_argument(
        '--use-agent-api',
        action='store_true',
        help='Use REST API for agent communication (requires: adk web --port 4200)'
    )
    parser.add_argument(
        '--agent-url',
        type=str,
        default='http://localhost:4200',
        help='Agent REST API URL (default: http://localhost:4200)'
    )
    parser.add_argument(
        '--no-fallback',
        action='store_true',
        help='Disable fallback to direct calls if agent API unavailable'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='validation',
        choices=['train', 'validation', 'test'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='Starting index for evaluation'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation/results/docvqa',
        help='Output directory for results'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from previous run'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Cache directory for HuggingFace dataset'
    )
    
    args = parser.parse_args()
    
    # Choose evaluator based on mode
    if args.use_agent_api:
        logger.info("Using REST API mode (v2 evaluator)")
        logger.info(f"Agent URL: {args.agent_url}")
        logger.info(f"Fallback enabled: {not args.no_fallback}")
        logger.info("\n⚠️  Make sure agent server is running: adk web --port 4200\n")
        
        # Initialize V2 evaluator (REST API)
        evaluator = DocVQAEvaluatorV2(
            agent_url=args.agent_url,
            enable_fallback=not args.no_fallback,
            dataset_split=args.split,
            cache_dir=args.cache_dir
        )
    else:
        logger.info("Using fallback mode (v1 evaluator - direct tool calls)")
        
        # Load your agent
        logger.info("Loading agent...")
        try:
            from agent import root_agent
            agent = root_agent
            logger.info(f"✓ Loaded agent: {agent.name}")
        except Exception as e:
            logger.error(f"Failed to load agent: {e}")
            logger.error("Make sure your agent is in agent.py")
            sys.exit(1)
        
        # Initialize V1 evaluator
        evaluator = DocVQAEvaluator(
            agent=agent,
            dataset_split=args.split,
            cache_dir=args.cache_dir
        )
    
    # Run evaluation
    logger.info("Starting evaluation...")
    metrics = evaluator.evaluate(
        num_samples=args.num_samples,
        start_idx=args.start_idx,
        output_dir=args.output_dir,
        resume=not args.no_resume
    )
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS")
    logger.info("="*80)
    logger.info(f"Overall ANLS: {metrics['overall_anls']:.4f} ({metrics['overall_anls']*100:.2f}%)")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"\nResults saved to: {args.output_dir}")
    logger.info("="*80)

if __name__ == "__main__":
    main()


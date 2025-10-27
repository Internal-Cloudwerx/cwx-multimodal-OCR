#!/usr/bin/env python3
"""
Multi-Agent DocVQA Evaluator

Enhanced evaluator that uses the complete multi-agent system:
- Orchestrator Agent for intelligent routing
- Specialist Agents (Vision, OCR, Layout)
- Answer Validator Agent for confidence scoring
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.hf_docvqa_loader import HFDocVQADataset
from evaluation.anls_metric import anls_score_multi
from typing import Dict, Optional
from tqdm import tqdm
import json
import pandas as pd
import time
import logging
from collections import defaultdict
import tempfile
import os
from dotenv import load_dotenv

# Import our multi-agent system
from agents.orchestrator import create_orchestrator
from agents.answer_validator import create_answer_validator

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiAgentDocVQAEvaluator:
    """
    Enhanced DocVQA evaluator using complete multi-agent system
    
    Features:
    - Intelligent routing via Orchestrator Agent
    - Specialist agents for different question types
    - Answer validation and confidence scoring
    - Comprehensive evaluation metrics
    """
    
    def __init__(
        self,
        dataset_split: str = "validation",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize multi-agent evaluator
        
        Args:
            dataset_split: 'train', 'validation', or 'test'
            cache_dir: Cache directory for dataset
        """
        self.dataset_split = dataset_split
        
        # Initialize multi-agent system
        logger.info("Initializing Multi-Agent System...")
        self.orchestrator = create_orchestrator()
        self.validator = create_answer_validator()
        
        # Load dataset
        logger.info("Loading DocVQA dataset...")
        self.loader = HFDocVQADataset(cache_dir=cache_dir)
        self.dataset = self.loader.load(dataset_split)
        
        self.results = []
        self.progress_file = None
        
        logger.info("Multi-Agent DocVQA Evaluator initialized successfully!")
    
    def evaluate(
        self,
        num_samples: Optional[int] = None,
        start_idx: int = 0,
        output_dir: str = "evaluation/results/docvqa_multi_agent",
        save_frequency: int = 10,
        resume: bool = True
    ) -> Dict:
        """
        Run evaluation using multi-agent system
        
        Args:
            num_samples: Number of samples to evaluate (None = all)
            start_idx: Starting index (for resuming)
            output_dir: Directory to save results
            save_frequency: Save progress every N samples
            resume: Resume from previous run if exists
            
        Returns:
            Dictionary with evaluation metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup progress file
        self.progress_file = output_path / f"progress_{self.dataset_split}.json"
        
        # Resume if requested
        if resume and self.progress_file.exists():
            start_idx, self.results = self._load_progress()
            logger.info(f"Resuming from sample {start_idx}")
        
        # Determine samples to process
        total_samples = len(self.dataset)
        end_idx = min(start_idx + num_samples, total_samples) if num_samples else total_samples
        
        logger.info(f"\n{'='*80}")
        logger.info(f"MULTI-AGENT EVALUATION STARTING")
        logger.info(f"{'='*80}")
        logger.info(f"Dataset: SP-DocVQA {self.dataset_split}")
        logger.info(f"Samples: {start_idx} to {end_idx} (total: {end_idx - start_idx})")
        logger.info(f"Multi-Agent System: Orchestrator + 3 Specialists + Validator")
        logger.info(f"{'='*80}\n")
        
        # Evaluate samples
        for idx in tqdm(range(start_idx, end_idx), desc="Multi-Agent Evaluation"):
            try:
                sample = self.loader.get_sample(idx)
                result = self._evaluate_sample_multi_agent(sample, idx)
                self.results.append(result)
                
                # Save progress periodically
                if (idx + 1) % save_frequency == 0:
                    self._save_progress(idx + 1)
                    
            except Exception as e:
                logger.error(f"Error on sample {idx}: {e}", exc_info=True)
                # Record error
                self.results.append({
                    'questionId': idx,
                    'error': str(e),
                    'success': False
                })
        
        # Compute final metrics
        metrics = self._compute_metrics()
        
        # Generate reports
        self._generate_reports(metrics, output_path)
        
        # Clean up progress file
        if self.progress_file.exists():
            self.progress_file.unlink()
        
        return metrics
    
    def _evaluate_sample_multi_agent(self, sample: Dict, idx: int) -> Dict:
        """Evaluate a single sample using multi-agent system"""
        
        question = sample['question']
        image = sample['image']  # PIL Image
        ground_truth_answers = sample['answers']
        question_types = sample.get('question_types', [])
        
        start_time = time.time()
        
        # Save image temporarily for agents
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            image.save(tmp_file.name, 'PNG')
            tmp_image_path = tmp_file.name
        
        try:
            # Step 1: Run Orchestrator Agent (intelligent routing)
            orchestrator_result = self.orchestrator.analyze_question(
                image_path=tmp_image_path,
                question=question,
                question_types=question_types
            )
            
            # Step 2: Validate answer with Answer Validator
            validation_result = self.validator.validate_answer(
                answer=orchestrator_result['answer'],
                question=question,
                question_types=question_types,
                ground_truth=ground_truth_answers,
                agent_result=orchestrator_result,
                image_path=tmp_image_path
            )
            
            # Step 3: Compute ANLS score
            anls = anls_score_multi(orchestrator_result['answer'], ground_truth_answers)
            
            processing_time = time.time() - start_time
            
            result = {
                'questionId': sample['questionId'],
                'question': question,
                'prediction': orchestrator_result['answer'],
                'ground_truth': ground_truth_answers,
                'anls_score': anls,
                'correct': anls >= 0.5,  # Binary correctness
                'question_types': question_types,
                'docId': sample['docId'],
                'processing_time': processing_time,
                'success': True,
                
                # Multi-agent specific data
                'orchestrator_result': orchestrator_result,
                'validation_result': validation_result,
                'selected_agent': orchestrator_result['routing']['selected_agent'],
                'routing_reason': orchestrator_result['routing']['routing_reason'],
                'agent_confidence': orchestrator_result['confidence'],
                'validator_confidence': validation_result['confidence_scores']['overall'],
                'ground_truth_match': validation_result['ground_truth_validation']['match'] if validation_result['ground_truth_validation'] else False,
                'ground_truth_similarity': validation_result['ground_truth_validation']['similarity'] if validation_result['ground_truth_validation'] else 0.0
            }
            
        except Exception as e:
            logger.error(f"Multi-agent system failed on sample {idx}: {e}")
            result = {
                'questionId': sample['questionId'],
                'question': question,
                'error': str(e),
                'success': False
            }
        
        finally:
            # Clean up temp file
            if os.path.exists(tmp_image_path):
                os.unlink(tmp_image_path)
        
        return result
    
    def _compute_metrics(self) -> Dict:
        """Compute comprehensive evaluation metrics"""
        if not self.results:
            return {"error": "No results to compute metrics"}
        
        successful_results = [r for r in self.results if r.get('success', False)]
        failed_results = [r for r in self.results if not r.get('success', False)]
        
        if not successful_results:
            return {"error": "No successful evaluations"}
        
        # Basic metrics
        total_samples = len(self.results)
        successful_samples = len(successful_results)
        failed_samples = len(failed_results)
        
        # ANLS metrics
        anls_scores = [r['anls_score'] for r in successful_results]
        avg_anls = sum(anls_scores) / len(anls_scores)
        
        # Accuracy metrics
        correct_predictions = sum(1 for r in successful_results if r['correct'])
        accuracy = correct_predictions / successful_samples
        
        # Processing time
        processing_times = [r['processing_time'] for r in successful_results]
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        # Multi-agent specific metrics
        agent_distribution = defaultdict(int)
        routing_accuracy = 0
        confidence_scores = []
        ground_truth_matches = 0
        
        for r in successful_results:
            if 'selected_agent' in r:
                agent_distribution[r['selected_agent']] += 1
            
            if 'agent_confidence' in r:
                confidence_scores.append(r['agent_confidence'])
            
            if 'ground_truth_match' in r and r['ground_truth_match']:
                ground_truth_matches += 1
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        ground_truth_accuracy = ground_truth_matches / successful_samples if successful_samples > 0 else 0
        
        # Performance by question type
        type_performance = defaultdict(lambda: {'total': 0, 'correct': 0, 'anls_scores': []})
        
        for r in successful_results:
            for q_type in r.get('question_types', []):
                type_performance[q_type]['total'] += 1
                if r['correct']:
                    type_performance[q_type]['correct'] += 1
                type_performance[q_type]['anls_scores'].append(r['anls_score'])
        
        # Calculate ANLS by type
        for q_type, data in type_performance.items():
            if data['anls_scores']:
                data['avg_anls'] = sum(data['anls_scores']) / len(data['anls_scores'])
                data['accuracy'] = data['correct'] / data['total']
            else:
                data['avg_anls'] = 0
                data['accuracy'] = 0
        
        metrics = {
            'total_samples': total_samples,
            'successful_samples': successful_samples,
            'failed_samples': failed_samples,
            'success_rate': successful_samples / total_samples,
            
            # Core metrics
            'avg_anls': avg_anls,
            'accuracy': accuracy,
            'avg_processing_time': avg_processing_time,
            
            # Multi-agent metrics
            'agent_distribution': dict(agent_distribution),
            'avg_confidence': avg_confidence,
            'ground_truth_accuracy': ground_truth_accuracy,
            
            # Performance by question type
            'type_performance': dict(type_performance),
            
            # System info
            'evaluator_type': 'multi_agent',
            'timestamp': time.time()
        }
        
        return metrics
    
    def _generate_reports(self, metrics: Dict, output_path: Path):
        """Generate comprehensive evaluation reports"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_file = output_path / f"detailed_results_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save metrics
        metrics_file = output_path / f"metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save CSV results
        csv_file = output_path / f"results_{timestamp}.csv"
        df_data = []
        for r in self.results:
            if r.get('success', False):
                df_data.append({
                    'questionId': r['questionId'],
                    'question': r['question'],
                    'prediction': r['prediction'],
                    'ground_truth': '; '.join(r['ground_truth']),
                    'anls_score': r['anls_score'],
                    'correct': r['correct'],
                    'question_types': '; '.join(r['question_types']),
                    'selected_agent': r.get('selected_agent', ''),
                    'agent_confidence': r.get('agent_confidence', 0),
                    'validator_confidence': r.get('validator_confidence', 0),
                    'ground_truth_match': r.get('ground_truth_match', False),
                    'processing_time': r['processing_time']
                })
        
        if df_data:
            df = pd.DataFrame(df_data)
            df.to_csv(csv_file, index=False)
        
        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info(f"MULTI-AGENT EVALUATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total Samples: {metrics['total_samples']}")
        logger.info(f"Successful: {metrics['successful_samples']} ({metrics['success_rate']:.1%})")
        logger.info(f"Average ANLS: {metrics['avg_anls']:.3f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.1%}")
        logger.info(f"Average Confidence: {metrics['avg_confidence']:.2f}")
        logger.info(f"Ground Truth Accuracy: {metrics['ground_truth_accuracy']:.1%}")
        logger.info(f"Average Processing Time: {metrics['avg_processing_time']:.2f}s")
        
        logger.info(f"\nAgent Distribution:")
        for agent, count in metrics['agent_distribution'].items():
            percentage = count / metrics['successful_samples'] * 100
            logger.info(f"  {agent}: {count} ({percentage:.1f}%)")
        
        logger.info(f"\nPerformance by Question Type:")
        for q_type, perf in metrics['type_performance'].items():
            logger.info(f"  {q_type}: {perf['accuracy']:.1%} accuracy, {perf['avg_anls']:.3f} ANLS ({perf['total']} samples)")
        
        logger.info(f"\nResults saved to: {output_path}")
        logger.info(f"{'='*80}")
    
    def _load_progress(self) -> tuple:
        """Load progress from file"""
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
            return data['last_idx'], data['results']
        except Exception as e:
            logger.warning(f"Failed to load progress: {e}")
            return 0, []
    
    def _save_progress(self, idx: int):
        """Save progress to file"""
        try:
            progress_data = {
                'last_idx': idx,
                'results': self.results,
                'timestamp': time.time()
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-Agent DocVQA Evaluator')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to evaluate')
    parser.add_argument('--start-idx', type=int, default=0, help='Starting index for evaluation')
    parser.add_argument('--output-dir', type=str, default='evaluation/results/docvqa_multi_agent_benchmark', help='Output directory for results')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh evaluation (don\'t resume)')
    args = parser.parse_args()
    
    # Test the multi-agent evaluator
    evaluator = MultiAgentDocVQAEvaluator()
    
    # Run evaluation with command line arguments
    metrics = evaluator.evaluate(
        num_samples=args.num_samples,
        start_idx=args.start_idx,
        output_dir=args.output_dir,
        resume=not args.no_resume
    )
    
    print("\nMulti-Agent Evaluation Test Complete!")
    print(f"Average ANLS: {metrics['avg_anls']:.3f}")
    print(f"Accuracy: {metrics['accuracy']:.1%}")
    print(f"Agent Distribution: {metrics['agent_distribution']}")

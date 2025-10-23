"""
DocVQA Benchmark Evaluator V2 - With REST API Agent Integration

This version uses the REST API approach for proper agent integration,
enabling multi-agent architecture development.

Usage:
    1. Start agent server: adk web --port 4200
    2. Run evaluation: python scripts/run_docvqa_benchmark.py --use-agent-api
"""

from evaluation.hf_docvqa_loader import HFDocVQADataset
from evaluation.anls_metric import anls_score_multi
from tools.agent_client import AgentClientManager
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import json
import pandas as pd
import time
import logging
from collections import defaultdict
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocVQAEvaluatorV2:
    """
    Evaluate agent on SP-DocVQA benchmark using REST API
    
    This evaluator properly integrates with ADK agents via REST API,
    enabling multi-agent orchestration and proper agent communication.
    """
    
    def __init__(
        self,
        agent_url: str = "http://localhost:4200",
        enable_fallback: bool = True,
        dataset_split: str = "validation",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize evaluator
        
        Args:
            agent_url: URL of ADK agent server (from `adk web`)
            enable_fallback: Whether to fall back to direct calls if API unavailable
            dataset_split: 'train', 'validation', or 'test'
            cache_dir: Cache directory for dataset
        """
        self.agent_url = agent_url
        self.enable_fallback = enable_fallback
        
        # Initialize agent client
        logger.info(f"Initializing agent client for {agent_url}...")
        self.agent_client = AgentClientManager(
            base_url=agent_url,
            enable_fallback=enable_fallback
        )
        
        # Rate limiting
        self.last_request_time = None
        self.min_request_interval = 2.0  # 2 seconds between requests
        
        # Load dataset
        logger.info("Loading DocVQA dataset...")
        self.loader = HFDocVQADataset(cache_dir=cache_dir)
        self.dataset = self.loader.load(dataset_split)
        
        self.results = []
        self.progress_file = None
        
    def evaluate(
        self,
        num_samples: Optional[int] = None,
        start_idx: int = 0,
        output_dir: str = "evaluation/results/docvqa",
        save_frequency: int = 10,
        resume: bool = True
    ) -> Dict:
        """
        Run evaluation on DocVQA benchmark
        
        Args:
            num_samples: Number of samples to evaluate (None = all)
            start_idx: Starting index in dataset
            output_dir: Directory to save results
            save_frequency: Save progress every N samples
            resume: Whether to resume from previous run
            
        Returns:
            Dict with evaluation metrics
        """
        # Determine evaluation range
        total_samples = len(self.dataset)
        if num_samples is None:
            num_samples = total_samples - start_idx
        else:
            num_samples = min(num_samples, total_samples - start_idx)
        
        end_idx = start_idx + num_samples
        
        logger.info(f"Evaluating samples {start_idx} to {end_idx-1} ({num_samples} total)")
        logger.info(f"Agent mode: {'REST API' if self.agent_client.api_available else 'Fallback'}")
        
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.progress_file = output_path / f"progress_{timestamp}.json"
        
        # Resume from previous run if requested
        start_from = start_idx
        if resume and self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                saved_progress = json.load(f)
                self.results = saved_progress.get('results', [])
                start_from = saved_progress.get('last_idx', start_idx) + 1
                logger.info(f"Resuming from sample {start_from}")
        
        # Evaluate samples
        for idx in tqdm(range(start_from, end_idx), desc="Evaluating"):
            try:
                result = self._evaluate_sample(idx)
                self.results.append(result)
                
                # Save progress periodically
                if (idx - start_idx + 1) % save_frequency == 0:
                    self._save_progress(idx, output_path, timestamp)
                    
            except Exception as e:
                logger.error(f"Error evaluating sample {idx}: {e}")
                # Add failed result
                sample = self.dataset[idx]
                self.results.append({
                    'sample_idx': idx,
                    'question_id': sample.get('questionId', f'q_{idx}'),
                    'question': sample['question'],
                    'predicted_answer': '',
                    'ground_truth': sample['answers'],
                    'anls_score': 0.0,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Final save
        self._save_progress(end_idx - 1, output_path, timestamp, final=True)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Save detailed results
        detailed_path = output_path / f"detailed_results_{timestamp}.json"
        with open(detailed_path, 'w') as f:
            json.dump({
                'evaluation_config': {
                    'agent_url': self.agent_url,
                    'agent_mode': 'REST API' if self.agent_client.api_available else 'Fallback',
                    'num_samples': num_samples,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                },
                'metrics': metrics,
                'results': self.results
            }, f, indent=2)
        
        # Save metrics
        metrics_path = output_path / f"metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save CSV
        csv_path = output_path / f"results_{timestamp}.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"\n{'='*60}")
        logger.info("EVALUATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Overall ANLS: {metrics['overall_anls']:.4f}")
        logger.info(f"Accuracy (ANLS ≥ 0.5): {metrics['accuracy']:.2f}%")
        logger.info(f"Total samples: {metrics['total_samples']}")
        logger.info(f"{'='*60}")
        logger.info(f"\nResults saved to:")
        logger.info(f"  - {detailed_path}")
        logger.info(f"  - {metrics_path}")
        logger.info(f"  - {csv_path}")
        
        return metrics
    
    def _evaluate_sample(self, idx: int) -> Dict:
        """Evaluate a single sample"""
        sample = self.dataset[idx]
        
        # Extract data
        question = sample['question']
        image = sample['image']  # PIL Image
        ground_truth_answers = sample['answers']
        question_types = sample.get('question_types', [])
        
        # Save image temporarily for agent
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            image.save(tmp_file.name, 'PNG')
            image_path = tmp_file.name
        
        try:
            # Rate limiting
            if self.last_request_time is not None:
                elapsed = time.time() - self.last_request_time
                if elapsed < self.min_request_interval:
                    time.sleep(self.min_request_interval - elapsed)
            
            # Run agent
            start_time = time.time()
            predicted_answer = self._run_agent(image_path, question, question_types)
            elapsed_time = time.time() - start_time
            
            self.last_request_time = time.time()
            
            # Calculate ANLS
            anls = anls_score_multi(predicted_answer, ground_truth_answers)
            
            # Normalize answer for better matching
            predicted_answer = self._normalize_answer(predicted_answer)
            
            # Recalculate ANLS with normalized answer
            anls = anls_score_multi(predicted_answer, ground_truth_answers)
            
            result = {
                'sample_idx': idx,
                'question_id': sample.get('questionId', f'q_{idx}'),
                'question': question,
                'question_types': question_types,
                'predicted_answer': predicted_answer,
                'ground_truth': ground_truth_answers,
                'anls_score': anls,
                'elapsed_time': elapsed_time,
                'status': 'success'
            }
            
            logger.info(
                f"Sample {idx}: ANLS={anls:.3f} | "
                f"Q: {question[:50]}... | "
                f"A: {predicted_answer[:50]}..."
            )
            
            return result
        finally:
            # Clean up temp file
            try:
                os.unlink(image_path)
            except:
                pass
    
    def _run_agent(self, image_path: str, question: str, question_types: list = None) -> str:
        """
        Run agent via REST API
        
        This properly uses the ADK agent through the REST API interface.
        """
        try:
            # Get OCR for structured questions
            ocr_context = None
            
            if question_types:
                ocr_beneficial_types = {
                    'table/list', 'figure/diagram', 'layout', 'form', 'free_text'
                }
                vision_only_types = {
                    'handwritten', 'Image/Photo', 'Yes/No'
                }
                
                has_vision_only = any(qt in vision_only_types for qt in question_types)
                has_ocr_beneficial = any(qt in ocr_beneficial_types for qt in question_types)
                use_ocr = has_ocr_beneficial and not has_vision_only
                
                if use_ocr:
                    from tools.document_ocr import process_document_with_ocr
                    try:
                        ocr_result = json.loads(process_document_with_ocr(image_path))
                        if ocr_result.get('status') != 'error':
                            ocr_context = ocr_result.get('full_text', '')[:1500]
                    except Exception as e:
                        logger.warning(f"OCR extraction failed: {e}")
            
            # Ask agent
            answer = self.agent_client.ask(
                question=question,
                image_path=image_path,
                context=ocr_context
            )
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Agent call failed: {e}")
            raise
    
    def _normalize_answer(self, answer: str) -> str:
        """Light normalization of answer"""
        if not answer:
            return ""
        
        # Basic cleanup
        answer = answer.strip()
        
        # Remove common prefixes
        prefixes = [
            "the answer is:",
            "answer:",
            "the answer:",
        ]
        
        answer_lower = answer.lower()
        for prefix in prefixes:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
                break
        
        # Remove quotes
        if (answer.startswith('"') and answer.endswith('"')) or \
           (answer.startswith("'") and answer.endswith("'")):
            answer = answer[1:-1].strip()
        
        return answer
    
    def _calculate_metrics(self) -> Dict:
        """Calculate evaluation metrics"""
        if not self.results:
            return {}
        
        # Overall metrics
        anls_scores = [r['anls_score'] for r in self.results if r.get('status') == 'success']
        total_samples = len(anls_scores)
        
        if total_samples == 0:
            return {'error': 'No successful evaluations'}
        
        overall_anls = sum(anls_scores) / total_samples
        accuracy = sum(1 for s in anls_scores if s >= 0.5) / total_samples * 100
        
        # Per question type
        type_metrics = defaultdict(lambda: {'scores': [], 'count': 0})
        
        for result in self.results:
            if result.get('status') != 'success':
                continue
                
            for qtype in result.get('question_types', ['unknown']):
                type_metrics[qtype]['scores'].append(result['anls_score'])
                type_metrics[qtype]['count'] += 1
        
        # Calculate per-type ANLS
        by_question_type = {}
        for qtype, data in type_metrics.items():
            if data['scores']:
                by_question_type[qtype] = {
                    'anls': sum(data['scores']) / len(data['scores']),
                    'count': data['count']
                }
        
        # Timing metrics
        elapsed_times = [r.get('elapsed_time', 0) for r in self.results if r.get('status') == 'success']
        avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0
        
        return {
            'overall_anls': overall_anls,
            'accuracy': accuracy,
            'total_samples': total_samples,
            'success_rate': len(anls_scores) / len(self.results) * 100,
            'avg_time_per_sample': avg_time,
            'by_question_type': by_question_type,
            'agent_mode': 'REST API' if self.agent_client.api_available else 'Fallback'
        }
    
    def _save_progress(self, last_idx: int, output_path: Path, timestamp: str, final: bool = False):
        """Save progress to file"""
        progress_data = {
            'last_idx': last_idx,
            'results': self.results,
            'timestamp': timestamp,
            'final': final
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)


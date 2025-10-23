"""
DocVQA Evaluation System
Benchmark document processing agent on SP-DocVQA dataset
"""

from .hf_docvqa_loader import HFDocVQADataset
from .anls_metric import anls_score, anls_score_multi, compute_anls_metrics
from .docvqa_evaluator import DocVQAEvaluator

__all__ = [
    'HFDocVQADataset',
    'anls_score',
    'anls_score_multi',
    'compute_anls_metrics',
    'DocVQAEvaluator'
]


"""
ANLS (Average Normalized Levenshtein Similarity) metric
Standard evaluation metric for Document VQA tasks
"""

import Levenshtein
from typing import List

def anls_score(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:
    """
    Compute ANLS score between prediction and ground truth
    
    ANLS = 1 - (normalized_edit_distance) if score >= threshold else 0
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        threshold: Minimum similarity threshold (default 0.5)
        
    Returns:
        ANLS score between 0 and 1
    """
    # Normalize strings
    pred = prediction.strip().lower()
    gt = ground_truth.strip().lower()
    
    # Handle empty strings
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    
    if len(pred) == 0:
        return 0.0
    
    # Compute edit distance
    edit_distance = Levenshtein.distance(pred, gt)
    
    # Normalize by max length
    max_length = max(len(pred), len(gt))
    normalized_distance = edit_distance / max_length
    
    # Compute similarity
    similarity = 1.0 - normalized_distance
    
    # Apply threshold
    return similarity if similarity >= threshold else 0.0


def anls_score_multi(prediction: str, ground_truths: List[str], threshold: float = 0.5) -> float:
    """
    Compute ANLS score with multiple ground truth answers
    Returns the maximum score across all ground truths
    
    Args:
        prediction: Predicted answer
        ground_truths: List of acceptable ground truth answers
        threshold: Minimum similarity threshold
        
    Returns:
        Maximum ANLS score
    """
    if not ground_truths:
        return 0.0
    
    scores = [anls_score(prediction, gt, threshold) for gt in ground_truths]
    return max(scores)


def compute_anls_metrics(predictions: List[str], ground_truths: List[List[str]]) -> dict:
    """
    Compute ANLS metrics for a batch of predictions
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answer lists
        
    Returns:
        Dictionary with ANLS statistics
    """
    assert len(predictions) == len(ground_truths), "Predictions and ground truths must have same length"
    
    scores = []
    for pred, gts in zip(predictions, ground_truths):
        score = anls_score_multi(pred, gts)
        scores.append(score)
    
    return {
        'anls': sum(scores) / len(scores) if scores else 0.0,
        'num_samples': len(scores),
        'scores': scores
    }


# Test the metric
if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("$45.67", ["$45.67"], 1.0),  # Exact match
        ("45.67", ["$45.67"], 0.8),   # Close match
        ("SuperMart", ["Super Mart"], 0.9),  # Minor difference
        ("wrong", ["correct"], 0.0),  # Below threshold
    ]
    
    print("ANLS Metric Tests:")
    print("="*80)
    for pred, gts, expected_min in test_cases:
        score = anls_score_multi(pred, gts)
        status = "✓" if score >= expected_min else "✗"
        print(f"{status} Prediction: '{pred}' | Ground Truth: {gts} | ANLS: {score:.3f}")


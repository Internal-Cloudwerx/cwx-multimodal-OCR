"""
ANLS (Average Normalized Levenshtein Similarity) metric
Standard evaluation metric for Document VQA tasks

Phase 1.1: Enhanced with smart answer comparison for number-word normalization
"""

import Levenshtein
import re
from typing import List

# Number to word mapping for common numbers (0-20, 30, 40, 50, 60, 70, 80, 90, 100)
NUMBER_WORD_MAP = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
    '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten',
    '11': 'eleven', '12': 'twelve', '13': 'thirteen', '14': 'fourteen', '15': 'fifteen',
    '16': 'sixteen', '17': 'seventeen', '18': 'eighteen', '19': 'nineteen', '20': 'twenty',
    '30': 'thirty', '40': 'forty', '50': 'fifty', '60': 'sixty', '70': 'seventy',
    '80': 'eighty', '90': 'ninety', '100': 'one hundred'
}

def _normalize_answer(answer: str) -> List[str]:
    """
    Normalize answer for smart comparison
    
    Returns multiple normalized versions to handle format variations:
    1. Original normalized (lowercase, stripped)
    2. Number-to-word conversion if applicable
    3. Word-to-number conversion if applicable
    4. Punctuation removal
    """
    # Basic normalization
    normalized = answer.strip().lower()
    
    # List to return all variations
    variations = [normalized]
    
    # Remove common punctuation for comparison
    no_punct = re.sub(r'[^\w\s]', '', normalized)
    if no_punct != normalized:
        variations.append(no_punct)
    
    # Number word conversion (bidirectional)
    for num, word in NUMBER_WORD_MAP.items():
        # If answer is just a number (e.g., "10"), add word version ("ten")
        if normalized == num or no_punct == num:
            variations.append(word)
            # Also add capitalized version
            variations.append(word.capitalize())
        
        # If answer is a word number (e.g., "ten"), add number version ("10")
        if normalized == word or normalized == word.capitalize():
            variations.append(num)
    
    return variations

def _smart_anls_score(pred_variations: List[str], gt_variations: List[str], threshold: float = 0.5) -> float:
    """
    Compute ANLS score using smart comparison across all variations
    
    Tries all combinations of prediction and ground truth variations
    Returns the best score found
    """
    best_score = 0.0
    
    for pred_var in pred_variations:
        for gt_var in gt_variations:
            if not pred_var or not gt_var:
                continue
            
            # Try exact match first (fast)
            if pred_var == gt_var:
                return 1.0
            
            # Compute edit distance
            edit_distance = Levenshtein.distance(pred_var, gt_var)
            max_length = max(len(pred_var), len(gt_var))
            
            if max_length == 0:
                continue
            
            normalized_distance = edit_distance / max_length
            similarity = 1.0 - normalized_distance
            
            if similarity >= threshold:
                best_score = max(best_score, similarity)
    
    return best_score

def anls_score(prediction: str, ground_truth: str, threshold: float = 0.5, use_smart_comparison: bool = True) -> float:
    """
    Compute ANLS score between prediction and ground truth
    Enhanced with smart comparison for Phase 1.1 (number-word normalization)
    
    ANLS = 1 - (normalized_edit_distance) if score >= threshold else 0
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        threshold: Minimum similarity threshold (default 0.5)
        use_smart_comparison: Use smart comparison for number-word normalization (default True)
        
    Returns:
        ANLS score between 0 and 1
    """
    # Handle empty strings
    if len(ground_truth.strip()) == 0:
        return 1.0 if len(prediction.strip()) == 0 else 0.0
    
    if len(prediction.strip()) == 0:
        return 0.0
    
    # Use smart comparison for number-word normalization (Phase 1.1)
    if use_smart_comparison:
        pred_variations = _normalize_answer(prediction)
        gt_variations = _normalize_answer(ground_truth)
        return _smart_anls_score(pred_variations, gt_variations, threshold)
    
    # Fallback to original simple comparison
    pred = prediction.strip().lower()
    gt = ground_truth.strip().lower()
    
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
        # Phase 1.1: Number-word normalization tests
        ("10", ["ten"], 0.8),  # Number to word - should match
        ("ten", ["10"], 0.8),  # Word to number - should match
        ("4", ["four"], 0.8),  # Single digit conversion
        ("four", ["4"], 0.8),  # Reverse conversion
    ]
    
    print("ANLS Metric Tests:")
    print("="*80)
    for pred, gts, expected_min in test_cases:
        score = anls_score_multi(pred, gts)
        status = "✓" if score >= expected_min else "✗"
        print(f"{status} Prediction: '{pred}' | Ground Truth: {gts} | ANLS: {score:.3f}")


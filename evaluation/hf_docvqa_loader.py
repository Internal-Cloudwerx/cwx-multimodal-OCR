"""
Load SP-DocVQA from Hugging Face datasets
Provides standard interface for DocVQA benchmark evaluation
"""

from datasets import load_dataset
from pathlib import Path
from typing import Dict, List, Optional
import json
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HFDocVQADataset:
    """
    Load and manage SP-DocVQA dataset from Hugging Face
    
    Dataset: lmms-lab/DocVQA (official DocVQA mirror)
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize dataset loader
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("evaluation/data/docvqa_hf")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = None
        
    def load(self, split: str = "validation"):
        """
        Load dataset from Hugging Face
        
        Args:
            split: 'train', 'validation', or 'test'
            
        Returns:
            Dataset object
        """
        logger.info(f"Loading SP-DocVQA {split} split from Hugging Face...")
        
        try:
            # Load dataset (will auto-download and cache)
            self.dataset = load_dataset(
                "lmms-lab/DocVQA",
                "DocVQA",  # Config name
                split=split,
                cache_dir=str(self.cache_dir)
            )
            
            logger.info(f"✓ Loaded {len(self.dataset)} samples")
            
            # Show dataset info
            self._show_dataset_info(split)
            
            return self.dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _show_dataset_info(self, split: str):
        """Display dataset statistics"""
        logger.info(f"\n{'='*80}")
        logger.info(f"SP-DOCVQA {split.upper()} SPLIT")
        logger.info(f"{'='*80}")
        logger.info(f"Total samples: {len(self.dataset)}")
        
        # Sample first item to show structure
        if len(self.dataset) > 0:
            sample = self.dataset[0]
            logger.info(f"\nSample structure:")
            logger.info(f"  Keys: {list(sample.keys())}")
            logger.info(f"  Question: {sample['question'][:80]}...")
            logger.info(f"  Answers: {sample['answers']}")
            logger.info(f"  Image: {type(sample['image'])}")
            
            # Count question types if available
            if 'question_types' in sample and sample['question_types']:
                from collections import Counter
                type_counts = Counter()
                for item in self.dataset:
                    if 'question_types' in item and item['question_types']:
                        for qtype in item['question_types']:
                            type_counts[qtype] += 1
                
                logger.info(f"\nQuestion type distribution:")
                for qtype, count in type_counts.most_common():
                    pct = (count / len(self.dataset)) * 100
                    logger.info(f"  {qtype}: {count} ({pct:.1f}%)")
        
        logger.info(f"{'='*80}\n")
    
    def get_sample(self, idx: int) -> Dict:
        """
        Get a single sample
        
        Returns:
            Dictionary with question, image, answers, metadata
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        sample = self.dataset[idx]
        
        return {
            'questionId': sample.get('questionId', idx),
            'question': sample['question'],
            'image': sample['image'],  # PIL Image
            'answers': sample['answers'],
            'question_types': sample.get('question_types', []),
            'docId': sample.get('doc_id', f"doc_{idx}"),
            'image_id': sample.get('image_id', f"img_{idx}")
        }
    
    def save_to_json(self, output_path: str, split: str = "validation"):
        """
        Save dataset to DocVQA JSON format for offline use
        
        Note: Images are NOT saved, only metadata
        """
        if self.dataset is None:
            self.load(split)
        
        docvqa_format = {
            "dataset_name": "SP-DocVQA",
            "dataset_split": split,
            "dataset_version": "1.0",
            "source": "Hugging Face (lmms-lab/DocVQA)",
            "num_samples": len(self.dataset),
            "data": []
        }
        
        logger.info(f"Converting {len(self.dataset)} samples to JSON format...")
        
        for idx, item in enumerate(self.dataset):
            entry = {
                "questionId": item.get('questionId', idx),
                "question": item['question'],
                "image_id": item.get('image_id', f"img_{idx}"),
                "docId": item.get('doc_id', f"doc_{idx}"),
                "answers": item['answers'],
                "question_types": item.get('question_types', [])
            }
            docvqa_format["data"].append(entry)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(docvqa_format, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved to: {output_file}")
        
        return output_file


# Quick test
if __name__ == "__main__":
    loader = HFDocVQADataset()
    
    # Load validation set
    dataset = loader.load("validation")
    
    # Get a sample
    sample = loader.get_sample(0)
    print(f"\nSample question: {sample['question']}")
    print(f"Ground truth answers: {sample['answers']}")
    print(f"Image type: {type(sample['image'])}")
    
    # Save to JSON
    loader.save_to_json("evaluation/data/docvqa_validation.json", "validation")


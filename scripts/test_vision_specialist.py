#!/usr/bin/env python3
"""
Test script for Vision Specialist Agent
Tests the agent on sample image questions
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.vision_specialist import create_vision_specialist
from evaluation.hf_docvqa_loader import HFDocVQADataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vision_specialist():
    """Test the Vision Specialist Agent on sample questions"""
    
    # Initialize agent
    logger.info("Initializing Vision Specialist Agent...")
    agent = create_vision_specialist()
    
    # Load a small sample of the dataset
    logger.info("Loading DocVQA dataset sample...")
    loader = HFDocVQADataset()
    dataset = loader.load("validation")
    
    # Filter for image/photo questions
    image_questions = []
    for i, sample in enumerate(dataset):
        question_types = sample.get('question_types', [])
        if 'Image/Photo' in question_types:
            image_questions.append((i, sample))
            if len(image_questions) >= 5:  # Test on 5 samples
                break
    
    logger.info(f"Found {len(image_questions)} image/photo questions to test")
    
    if not image_questions:
        logger.warning("No image/photo questions found in dataset")
        return
    
    # Test each question
    results = []
    for idx, (sample_idx, sample) in enumerate(image_questions):
        logger.info(f"\n--- Test {idx + 1}/{len(image_questions)} ---")
        logger.info(f"Question: {sample['question']}")
        logger.info(f"Question Types: {sample.get('question_types', [])}")
        logger.info(f"Ground Truth: {sample['answers']}")
        
        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            sample['image'].save(tmp_file.name, 'PNG')
            image_path = tmp_file.name
        
        try:
            # Run Vision Specialist
            result = agent.analyze_image_question(
                image_path=image_path,
                question=sample['question'],
                question_types=sample.get('question_types', [])
            )
            
            logger.info(f"Answer: {result['answer']}")
            logger.info(f"Confidence: {result['confidence']:.2f}")
            logger.info(f"Processing Time: {result['processing_time']:.2f}s")
            logger.info(f"Status: {result['status']}")
            
            results.append({
                'sample_idx': sample_idx,
                'question': sample['question'],
                'question_types': sample.get('question_types', []),
                'ground_truth': sample['answers'],
                'predicted_answer': result['answer'],
                'confidence': result['confidence'],
                'processing_time': result['processing_time'],
                'status': result['status']
            })
            
        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {e}")
            results.append({
                'sample_idx': sample_idx,
                'question': sample['question'],
                'question_types': sample.get('question_types', []),
                'ground_truth': sample['answers'],
                'predicted_answer': "",
                'confidence': 0.0,
                'processing_time': 0.0,
                'status': 'error',
                'error': str(e)
            })
        
        finally:
            # Cleanup
            os.unlink(image_path)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("VISION SPECIALIST AGENT TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    successful_tests = [r for r in results if r['status'] == 'success']
    avg_confidence = sum(r['confidence'] for r in successful_tests) / len(successful_tests) if successful_tests else 0
    avg_time = sum(r['processing_time'] for r in successful_tests) / len(successful_tests) if successful_tests else 0
    
    logger.info(f"Total Tests: {len(results)}")
    logger.info(f"Successful: {len(successful_tests)}")
    logger.info(f"Failed: {len(results) - len(successful_tests)}")
    logger.info(f"Average Confidence: {avg_confidence:.2f}")
    logger.info(f"Average Processing Time: {avg_time:.2f}s")
    
    # Show sample results
    logger.info(f"\nSample Results:")
    for i, result in enumerate(results[:3]):
        logger.info(f"{i+1}. Q: {result['question'][:50]}...")
        logger.info(f"   A: {result['predicted_answer'][:50]}...")
        logger.info(f"   GT: {result['ground_truth']}")
        logger.info(f"   Conf: {result['confidence']:.2f}")
    
    return results

if __name__ == "__main__":
    test_vision_specialist()

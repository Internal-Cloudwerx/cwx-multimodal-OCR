#!/usr/bin/env python3
"""
Test Layout Specialist Agent
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.layout_specialist import create_layout_specialist
from evaluation.hf_docvqa_loader import HFDocVQADataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_layout_specialist():
    """Test Layout Specialist Agent on layout questions"""
    
    # Initialize agent
    logger.info("Initializing Layout Specialist Agent...")
    agent = create_layout_specialist()
    
    # Load dataset
    logger.info("Loading DocVQA dataset sample...")
    loader = HFDocVQADataset()
    dataset = loader.load("validation")
    
    # Filter for layout questions
    layout_questions = []
    for i, sample in enumerate(dataset):
        question_types = sample.get('question_types', [])
        if 'layout' in question_types:
            layout_questions.append((i, sample))
            if len(layout_questions) >= 5:  # Test 5 layout questions
                break
    
    if not layout_questions:
        logger.warning("No layout questions found")
        return
    
    logger.info(f"Found {len(layout_questions)} layout questions to test")
    
    # Test each question
    results = []
    for i, (sample_idx, sample) in enumerate(layout_questions):
        logger.info(f"\n--- Test {i+1}/{len(layout_questions)} ---")
        logger.info(f"Question: {sample['question']}")
        logger.info(f"Question Types: {sample.get('question_types', [])}")
        logger.info(f"Ground Truth: {sample['answers']}")
        
        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            sample['image'].save(tmp_file.name, 'PNG')
            image_path = tmp_file.name
        
        try:
            # Run Layout Specialist
            result = agent.analyze_layout_question(
                image_path=image_path,
                question=sample['question'],
                question_types=sample.get('question_types', [])
            )
            
            logger.info(f"Answer: {result['answer']}")
            logger.info(f"Confidence: {result['confidence']:.2f}")
            logger.info(f"OCR Quality: {result.get('ocr_data_quality', 'unknown')}")
            logger.info(f"Processing Time: {result['processing_time']:.2f}s")
            logger.info(f"Status: {result['status']}")
            
            results.append({
                'question': sample['question'],
                'answer': result['answer'],
                'ground_truth': sample['answers'],
                'confidence': result['confidence'],
                'ocr_quality': result.get('ocr_data_quality', 'unknown'),
                'processing_time': result['processing_time'],
                'status': result['status']
            })
            
        except Exception as e:
            logger.error(f"Error: {e}")
            results.append({
                'question': sample['question'],
                'answer': f"Error: {str(e)}",
                'ground_truth': sample['answers'],
                'confidence': 0.0,
                'ocr_quality': 'unknown',
                'processing_time': 0.0,
                'status': 'error'
            })
        
        finally:
            # Cleanup
            os.unlink(image_path)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("LAYOUT SPECIALIST AGENT TEST SUMMARY")
    logger.info("="*60)
    
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    logger.info(f"Total Tests: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    if successful > 0:
        avg_confidence = sum(r['confidence'] for r in results if r['status'] == 'success') / successful
        avg_time = sum(r['processing_time'] for r in results if r['status'] == 'success') / successful
        logger.info(f"Average Confidence: {avg_confidence:.2f}")
        logger.info(f"Average Processing Time: {avg_time:.2f}s")
        
        # OCR Quality distribution
        ocr_qualities = {}
        for r in results:
            if r['status'] == 'success':
                quality = r['ocr_quality']
                ocr_qualities[quality] = ocr_qualities.get(quality, 0) + 1
        logger.info(f"OCR Quality Distribution: {ocr_qualities}")
    
    logger.info("\nSample Results:")
    for i, result in enumerate(results[:3]):  # Show first 3 results
        logger.info(f"{i+1}. Q: {result['question'][:50]}...")
        logger.info(f"   A: {result['answer'][:50]}...")
        logger.info(f"   GT: {result['ground_truth']}")
        logger.info(f"   Conf: {result['confidence']:.2f}, OCR: {result['ocr_quality']}")

if __name__ == "__main__":
    test_layout_specialist()

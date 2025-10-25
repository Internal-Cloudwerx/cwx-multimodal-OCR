#!/usr/bin/env python3
"""
Test Orchestrator Agent - Multi-Agent Routing System
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.orchestrator import create_orchestrator
from evaluation.hf_docvqa_loader import HFDocVQADataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_orchestrator_routing():
    """Test Orchestrator Agent routing to different specialist agents"""
    
    # Initialize orchestrator
    logger.info("Initializing Orchestrator Agent...")
    orchestrator = create_orchestrator()
    
    # Load dataset
    logger.info("Loading DocVQA dataset sample...")
    loader = HFDocVQADataset()
    dataset = loader.load("validation")
    
    # Test different question types to verify routing
    test_cases = []
    
    # Find one example of each major question type
    question_type_samples = {}
    for sample in dataset:
        question_types = sample.get('question_types', [])
        for qt in question_types:
            if qt not in question_type_samples:
                question_type_samples[qt] = sample
                if len(question_type_samples) >= 8:  # Test 8 different types
                    break
        if len(question_type_samples) >= 8:
            break
    
    logger.info(f"Found {len(question_type_samples)} different question types to test")
    
    # Test each question type
    results = []
    for i, (question_type, sample) in enumerate(question_type_samples.items()):
        logger.info(f"\n--- Test {i+1}/{len(question_type_samples)} ---")
        logger.info(f"Question Type: {question_type}")
        logger.info(f"Question: {sample['question']}")
        logger.info(f"Ground Truth: {sample['answers']}")
        
        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            sample['image'].save(tmp_file.name, 'PNG')
            image_path = tmp_file.name
        
        try:
            # Run Orchestrator
            result = orchestrator.analyze_question(
                image_path=image_path,
                question=sample['question'],
                question_types=sample.get('question_types', [])
            )
            
            selected_agent = result['routing']['selected_agent']
            routing_reason = result['routing']['routing_reason']
            
            logger.info(f"Selected Agent: {selected_agent}")
            logger.info(f"Routing Reason: {routing_reason}")
            logger.info(f"Answer: {result['answer']}")
            logger.info(f"Confidence: {result['confidence']:.2f}")
            logger.info(f"Processing Time: {result['processing_time']:.2f}s")
            logger.info(f"Status: {result['status']}")
            
            results.append({
                'question_type': question_type,
                'question': sample['question'],
                'answer': result['answer'],
                'ground_truth': sample['answers'],
                'selected_agent': selected_agent,
                'routing_reason': routing_reason,
                'confidence': result['confidence'],
                'processing_time': result['processing_time'],
                'status': result['status']
            })
            
        except Exception as e:
            logger.error(f"Error: {e}")
            results.append({
                'question_type': question_type,
                'question': sample['question'],
                'answer': f"Error: {str(e)}",
                'ground_truth': sample['answers'],
                'selected_agent': 'error',
                'routing_reason': f"Error: {str(e)}",
                'confidence': 0.0,
                'processing_time': 0.0,
                'status': 'error'
            })
        
        finally:
            # Cleanup
            os.unlink(image_path)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("ORCHESTRATOR AGENT TEST SUMMARY")
    logger.info("="*80)
    
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
        
        # Agent distribution
        agent_distribution = {}
        for r in results:
            if r['status'] == 'success':
                agent = r['selected_agent']
                agent_distribution[agent] = agent_distribution.get(agent, 0) + 1
        logger.info(f"Agent Distribution: {agent_distribution}")
        
        # Routing accuracy - check if routing follows priority-based strategy
        routing_correct = 0
        total_routable = 0
        
        for r in results:
            if r['status'] == 'success':
                question_type = r['question_type']
                selected_agent = r['selected_agent']
                
                # Priority-based expected routing (OCR has highest priority)
                priority_routing = {
                    # Priority 1: OCR Specialist (highest performance)
                    'table/list': 'ocr',
                    'form': 'ocr', 
                    'free_text': 'ocr',
                    'others': 'ocr',
                    
                    # Priority 2: Layout Specialist (good performance)
                    'layout': 'layout',
                    
                    # Priority 3: Vision Specialist (improvement needed)
                    'Image/Photo': 'vision',
                    'figure/diagram': 'vision',
                    'handwritten': 'vision',
                    'Yes/No': 'vision'
                }
                
                if question_type in priority_routing:
                    total_routable += 1
                    if selected_agent == priority_routing[question_type]:
                        routing_correct += 1
        
        routing_accuracy = routing_correct / total_routable if total_routable > 0 else 0
        logger.info(f"Routing Accuracy: {routing_accuracy:.2%}")
    
    logger.info("\nSample Results:")
    for i, result in enumerate(results[:5]):  # Show first 5 results
        logger.info(f"{i+1}. Type: {result['question_type']}")
        logger.info(f"   Q: {result['question'][:50]}...")
        logger.info(f"   Agent: {result['selected_agent']} ({result['routing_reason']})")
        logger.info(f"   A: {result['answer'][:50]}...")
        logger.info(f"   Conf: {result['confidence']:.2f}")
    
    # Show orchestrator capabilities
    logger.info("\n" + "="*80)
    logger.info("ORCHESTRATOR CAPABILITIES")
    logger.info("="*80)
    
    capabilities = orchestrator.get_agent_capabilities()
    logger.info(f"Orchestrator: {capabilities['orchestrator']['name']}")
    logger.info(f"Description: {capabilities['orchestrator']['description']}")
    
    logger.info("\nSpecialist Agents:")
    for agent_name, info in capabilities['specialists'].items():
        logger.info(f"  {agent_name.upper()}: {info['name']}")
        logger.info(f"    Specialties: {info['specialties']}")
    
    # Show routing strategy
    logger.info("\nRouting Strategy:")
    routing_stats = orchestrator.get_routing_stats()
    for question_type, agent in routing_stats['routing_strategy'].items():
        logger.info(f"  {question_type} → {agent}")

if __name__ == "__main__":
    test_orchestrator_routing()

#!/usr/bin/env python3
"""
ADK A2A Demo Script
Demonstrates rich agent-to-agent communication for document analysis
"""

import sys
import tempfile
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

sys.path.insert(0, str(project_root.parent))
from adk_a2a_wrapper import create_adk_a2a_wrapper
from evaluation.hf_docvqa_loader import HFDocVQADataset

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_a2a_communication():
    """Demo A2A communication with different question types"""
    
    print("🚀 ADK A2A Multi-Agent Communication Demo")
    print("=" * 60)
    
    # Initialize A2A wrapper
    wrapper = create_adk_a2a_wrapper()
    
    # Load sample data
    loader = HFDocVQADataset()
    dataset = loader.load("validation")
    
    # Demo scenarios
    demo_scenarios = [
        {
            "name": "Vision Specialist Demo",
            "description": "Image/Photo question routing",
            "question_types": ["Image/Photo"],
            "sample_idx": 0
        },
        {
            "name": "OCR Specialist Demo", 
            "description": "Table/Form question routing",
            "question_types": ["table/list"],
            "sample_idx": 1
        },
        {
            "name": "Layout Specialist Demo",
            "description": "Layout question routing", 
            "question_types": ["layout"],
            "sample_idx": 2
        },
        {
            "name": "Low Confidence Collaboration",
            "description": "A2A collaboration when confidence is low",
            "question_types": ["free_text"],
            "sample_idx": 3
        }
    ]
    
    for i, scenario in enumerate(demo_scenarios, 1):
        print(f"\n🎬 Demo {i}: {scenario['name']}")
        print(f"📝 {scenario['description']}")
        print("-" * 40)
        
        # Find appropriate sample
        sample = None
        for s in dataset:
            if any(qt in s.get('question_types', []) for qt in scenario['question_types']):
                sample = s
                break
        
        if not sample:
            print(f"❌ No sample found for {scenario['question_types']}")
            continue
        
        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            sample['image'].save(tmp_file.name, 'PNG')
            image_path = tmp_file.name
        
        try:
            # Run A2A analysis
            result = wrapper.analyze_document_with_a2a(
                image_path=image_path,
                question=sample['question'],
                question_types=sample.get('question_types', []),
                show_communication=True
            )
            
            if result is None:
                print(f"❌ A2A analysis returned None")
                continue
            
            # Show results
            print(f"\n📊 Results:")
            print(f"  Answer: {result.get('answer', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 0.0):.2f}")
            print(f"  Processing Time: {result.get('processing_time', 0.0):.2f}s")
            print(f"  Selected Agent: {result.get('routing', {}).get('selected_agent', 'N/A')}")
            print(f"  A2A Steps: {result.get('a2a_communication', {}).get('total_steps', 0)}")
            print(f"  Collaboration Triggered: {result.get('a2a_communication', {}).get('collaboration_triggered', False)}")
            
            # Show communication log
            a2a_comm = result.get('a2a_communication', {})
            if a2a_comm.get('communication_log'):
                print(f"\n💬 A2A Communication Log:")
                for log_entry in a2a_comm['communication_log']:
                    print(f"  Step {log_entry.get('step', 'N/A')}: {log_entry.get('agent', 'N/A')} - {log_entry.get('action', 'N/A')}")
                    print(f"    {log_entry.get('message', 'N/A')}")
                    if 'decision' in log_entry:
                        print(f"    Decision: {log_entry['decision']}")
            
            # Show collaboration log if any
            if a2a_comm.get('collaboration_log'):
                print(f"\n🤝 Collaboration Log:")
                for log_entry in a2a_comm['collaboration_log']:
                    print(f"  {log_entry.get('agent', 'N/A')} - {log_entry.get('action', 'N/A')}")
                    print(f"    {log_entry.get('message', 'N/A')}")
        
        except Exception as e:
            print(f"❌ Demo failed: {e}")
        
        finally:
            # Cleanup
            import os
            os.unlink(image_path)
        
        print("\n" + "="*60)

def demo_a2a_capabilities():
    """Show A2A wrapper capabilities"""
    
    print("\n🔧 ADK A2A Wrapper Capabilities")
    print("=" * 40)
    
    wrapper = create_adk_a2a_wrapper()
    demo_info = wrapper.get_a2a_demo_info()
    
    print(f"📋 {demo_info['name']}")
    print(f"📝 {demo_info['description']}")
    print(f"🔧 Version: {demo_info['version']}")
    
    print(f"\n🤖 Available Agents:")
    for agent_name, agent_info in demo_info['agents'].items():
        print(f"  {agent_name}: {agent_info.get('description', 'Specialist agent')}")
        if 'specialties' in agent_info:
            print(f"    Specialties: {', '.join(agent_info['specialties'])}")
    
    print(f"\n✨ Capabilities:")
    for capability in demo_info['capabilities']:
        print(f"  • {capability}")

if __name__ == "__main__":
    # Show capabilities
    demo_a2a_capabilities()
    
    # Run communication demos
    demo_a2a_communication()
    
    print("\n🎉 ADK A2A Demo Complete!")
    print("Ready for enterprise demos and presentations! 🚀")

#!/usr/bin/env python3
"""
Orchestrator Agent

Intelligent routing system that coordinates specialist agents based on question types
and implements the optimized multi-agent strategy.
"""

import os
import time
import re
import logging
import warnings
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Import specialist agents
from agents.vision_specialist import create_vision_specialist
from agents.ocr_specialist import create_ocr_specialist
from agents.layout_specialist import create_layout_specialist

warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")
load_dotenv()
logger = logging.getLogger(__name__)

class OrchestratorAgent:
    """
    Orchestrator Agent for intelligent routing to specialist agents
    
    Implements the optimized routing strategy:
    - Vision Specialist: Image/Photo, figure/diagram, handwritten, Yes/No
    - OCR Specialist: table/list, form, free_text, others
    - Layout Specialist: layout, others (fallback)
    """
    
    def __init__(self):
        """
        Initialize Orchestrator Agent with all specialist agents
        """
        # Initialize specialist agents
        self.vision_specialist = create_vision_specialist()
        self.ocr_specialist = create_ocr_specialist()
        self.layout_specialist = create_layout_specialist()
        
        # Agent registry for easy access
        self.agents = {
            'vision': self.vision_specialist,
            'ocr': self.ocr_specialist,
            'layout': self.layout_specialist
        }
        
        # Routing strategy based on performance data
        self.routing_strategy = {
            # Vision Specialist specialties (highest impact)
            'Image/Photo': 'vision',      # 65% → 80-85% improvement
            'figure/diagram': 'vision',   # 75% → better with visual analysis
            'handwritten': 'vision',      # Visual analysis needed
            'Yes/No': 'vision',          # Simple visual questions
            
            # OCR Specialist specialties (excellent performance)
            'table/list': 'ocr',         # 99.52% ANLS - excellent
            'form': 'ocr',               # 89.98% ANLS - very good
            'free_text': 'ocr',          # Text-heavy questions
            'others': 'ocr',             # Generic questions benefit from OCR
            
            # Layout Specialist specialties (good performance)
            'layout': 'layout',          # 95.78% ANLS - excellent
        }
        
        logger.info("Orchestrator Agent initialized with 3 specialist agents")
        logger.info(f"Routing strategy: {len(self.routing_strategy)} question types mapped")
    
    def analyze_question(
        self, 
        image_path: str, 
        question: str, 
        question_types: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze question using intelligent routing to appropriate specialist
        
        Args:
            image_path: Path to the document image
            question: Question to answer
            question_types: List of question types
            
        Returns:
            Dictionary with answer, confidence, routing info, and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Determine best agent for this question
            selected_agent, routing_reason = self._select_agent(question_types, question)
            
            # Step 2: Route to selected agent
            logger.info(f"Routing to {selected_agent} agent: {routing_reason}")
            
            agent_result = self._route_to_agent(
                agent_name=selected_agent,
                image_path=image_path,
                question=question,
                question_types=question_types
            )
            
            # Step 3: Enhance result with routing metadata
            total_time = time.time() - start_time
            
            result = {
                **agent_result,
                "routing": {
                    "selected_agent": selected_agent,
                    "routing_reason": routing_reason,
                    "question_types": question_types or [],
                    "routing_time": total_time - agent_result.get('processing_time', 0)
                },
                "orchestrator_version": "1.0.0"
            }
            
            logger.info(f"Orchestrator: '{agent_result.get('answer', '')[:50]}...' via {selected_agent} (confidence: {agent_result.get('confidence', 0):.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Orchestrator failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "status": "error",
                "agent_type": "orchestrator",
                "routing": {
                    "selected_agent": "none",
                    "routing_reason": f"Error: {str(e)}",
                    "question_types": question_types or []
                },
                "error": str(e)
            }
    
    def _detect_question_type(self, question: str) -> List[str]:
        """
        Intelligently detect question type from question text
        
        Args:
            question: The question text
            
        Returns:
            List of detected question types
        """
        question_lower = question.lower()
        detected_types = []
        
        # Check Yes/No questions FIRST (before other patterns)
        yes_no_pattern = r'^(is there|are there|does (this|it|that)|do you|can you|will you|was there|were there|is this|are you|is it)(\s|$)'
        if re.search(yes_no_pattern, question_lower):
            detected_types.append('Yes/No')
            return detected_types  # Yes/No is very specific, return early
        
        # Vision/Image indicators
        vision_keywords = [
            r'\b(what|describe|identify|show|see).*\b(image|photo|picture|diagram|chart|graph|figure|illustration)',
            r'\b(caption|title|label).*\b(image|figure|diagram)',
            r'(how many|count).*\b(image|figure|photo)',
            r'(handwritten|hand writing)',
        ]
        
        for pattern in vision_keywords:
            if re.search(pattern, question_lower):
                detected_types.append('Image/Photo')
                break
        
        # Layout indicators  
        layout_keywords = [
            r'\b(structure|layout|organization|arrangement|organization)',
            r'\b(heading|title|caption|section)',
            r'(how is|what is the structure|what is the layout)',
            r'(where is|which section)',
        ]
        
        for pattern in layout_keywords:
            if re.search(pattern, question_lower):
                detected_types.append('layout')
                break
        
        # OCR indicators (tables, forms, structured data)
        ocr_keywords = [
            r'\b(table|list|row|column|cell)',
            r'\b(form|field|signature|date|name|address)',
            r'(what is the total|how much|cost|price|amount)',
            r'(extract|extraction)',
        ]
        
        for pattern in ocr_keywords:
            if re.search(pattern, question_lower):
                detected_types.append('table/list')
                break
        
        # Default to OCR if nothing detected
        if not detected_types:
            detected_types = ['others']
        
        return list(set(detected_types))  # Remove duplicates
    
    def _select_agent(self, question_types: Optional[List[str]], question: str = "") -> Tuple[str, str]:
        """
        Select the best agent for the given question types
        Auto-detects question type if not provided
        
        Args:
            question_types: List of question types (optional)
            question: Question text for auto-detection
            
        Returns:
            Tuple of (agent_name, routing_reason)
        """
        # If no question types provided, try to detect them
        if not question_types or question_types == []:
            if question:
                detected_types = self._detect_question_type(question)
                logger.info(f"Auto-detected question types: {detected_types}")
                question_types = detected_types
            else:
                return 'ocr', "No question types specified and no question provided, defaulting to OCR specialist"
        
        # Priority-based routing: Check question types in order of performance impact
        # This ensures we route to the best agent even when multiple types are present
        
        # Priority 1: OCR Specialist (highest performance)
        ocr_priority_types = ['table/list', 'form', 'free_text', 'others']
        for question_type in question_types:
            if question_type in ocr_priority_types:
                return 'ocr', f"Question type '{question_type}' routed to OCR specialist (priority routing)"
        
        # Priority 2: Layout Specialist (good performance)
        layout_priority_types = ['layout']
        for question_type in question_types:
            if question_type in layout_priority_types:
                return 'layout', f"Question type '{question_type}' routed to Layout specialist (priority routing)"
        
        # Priority 3: Vision Specialist (improvement needed)
        vision_priority_types = ['Image/Photo', 'figure/diagram', 'handwritten', 'Yes/No']
        for question_type in question_types:
            if question_type in vision_priority_types:
                return 'vision', f"Question type '{question_type}' routed to Vision specialist (priority routing)"
        
        # Fallback: Use OCR specialist for unknown question types
        return 'ocr', f"Unknown question types {question_types}, defaulting to OCR specialist"
    
    def _route_to_agent(
        self, 
        agent_name: str, 
        image_path: str, 
        question: str, 
        question_types: Optional[List[str]]
    ) -> Dict:
        """
        Route question to the specified agent
        
        Args:
            agent_name: Name of the agent to use
            image_path: Path to the document image
            question: Question to answer
            question_types: List of question types
            
        Returns:
            Agent result dictionary
        """
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        # Route to appropriate agent method
        if agent_name == 'vision':
            return agent.analyze_image_question(image_path, question, question_types)
        elif agent_name == 'ocr':
            return agent.analyze_table_form_question(image_path, question, question_types)
        elif agent_name == 'layout':
            return agent.analyze_layout_question(image_path, question, question_types)
        else:
            raise ValueError(f"No method defined for agent: {agent_name}")
    
    def get_agent_capabilities(self) -> Dict:
        """
        Get capabilities of all agents
        
        Returns:
            Dictionary with agent capabilities
        """
        capabilities = {}
        
        for agent_name, agent in self.agents.items():
            capabilities[agent_name] = agent.get_agent_info()
        
        return {
            "orchestrator": {
                "name": "Orchestrator Agent",
                "description": "Intelligent routing system for multi-agent document analysis",
                "version": "1.0.0",
                "routing_strategy": self.routing_strategy
            },
            "specialists": capabilities
        }
    
    def get_routing_stats(self) -> Dict:
        """
        Get routing statistics and strategy information
        
        Returns:
            Dictionary with routing statistics
        """
        return {
            "total_question_types": len(self.routing_strategy),
            "routing_strategy": self.routing_strategy,
            "agent_distribution": {
                agent: len([qt for qt, a in self.routing_strategy.items() if a == agent])
                for agent in ['vision', 'ocr', 'layout']
            },
            "performance_targets": {
                "Image/Photo": "65% → 80-85% (Vision Specialist)",
                "figure/diagram": "75% → better (Vision Specialist)", 
                "table/list": "99.52% ANLS (OCR Specialist)",
                "form": "89.98% ANLS (OCR Specialist)",
                "layout": "95.78% ANLS (Layout Specialist)"
            }
        }


def create_orchestrator() -> OrchestratorAgent:
    """
    Create an Orchestrator Agent instance
    
    Returns:
        OrchestratorAgent instance
    """
    return OrchestratorAgent()


if __name__ == "__main__":
    # Test the Orchestrator Agent
    import tempfile
    from PIL import Image
    
    # Create a test image
    test_image = Image.new('RGB', (100, 100), color='white')
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        test_image.save(tmp_file.name, 'PNG')
        test_path = tmp_file.name
    
    try:
        # Initialize orchestrator
        orchestrator = create_orchestrator()
        
        # Test different question types
        test_cases = [
            ("What is in this image?", ["Image/Photo"]),
            ("What is the total amount?", ["table/list"]),
            ("What is the company name?", ["layout"]),
            ("Unknown question type", ["unknown_type"])
        ]
        
        print("Orchestrator Agent Test Results:")
        print("=" * 50)
        
        for question, question_types in test_cases:
            result = orchestrator.analyze_question(
                image_path=test_path,
                question=question,
                question_types=question_types
            )
            
            print(f"\nQuestion: {question}")
            print(f"Types: {question_types}")
            print(f"Selected Agent: {result['routing']['selected_agent']}")
            print(f"Reason: {result['routing']['routing_reason']}")
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Status: {result['status']}")
        
        # Show capabilities
        print("\n" + "=" * 50)
        print("Agent Capabilities:")
        capabilities = orchestrator.get_agent_capabilities()
        for agent_name, info in capabilities['specialists'].items():
            print(f"\n{agent_name.upper()}:")
            print(f"  Name: {info['name']}")
            print(f"  Specialties: {info['specialties']}")
        
        # Show routing stats
        print("\n" + "=" * 50)
        print("Routing Statistics:")
        stats = orchestrator.get_routing_stats()
        print(f"Total Question Types: {stats['total_question_types']}")
        print(f"Agent Distribution: {stats['agent_distribution']}")
        
    finally:
        # Cleanup
        os.unlink(test_path)

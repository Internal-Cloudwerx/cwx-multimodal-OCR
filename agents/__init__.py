"""
Multi-Agent System for Document Analysis

This package contains specialized agents for different types of document questions:
- Vision Specialist: Image/photo questions
- OCR Specialist: Table/form questions  
- Layout Specialist: Layout/abstract questions
- Orchestrator: Routes questions to appropriate specialists
- Answer Validator: Validates and scores answers
"""

from .vision_specialist import VisionSpecialistAgent, create_vision_specialist
from .ocr_specialist import OCRSpecialistAgent, create_ocr_specialist
from .layout_specialist import LayoutSpecialistAgent, create_layout_specialist
from .orchestrator import OrchestratorAgent, create_orchestrator
from .answer_validator import AnswerValidatorAgent, create_answer_validator

__all__ = [
    'VisionSpecialistAgent',
    'create_vision_specialist',
    'OCRSpecialistAgent',
    'create_ocr_specialist',
    'LayoutSpecialistAgent',
    'create_layout_specialist',
    'OrchestratorAgent',
    'create_orchestrator',
    'AnswerValidatorAgent',
    'create_answer_validator'
]

#!/usr/bin/env python3
"""
Layout Specialist Agent

Specialized in layout and abstract questions using hybrid OCR + Vision approach
for comprehensive document understanding and spatial reasoning.
"""

import os
import time
import logging
import warnings
import json
import tempfile
from typing import Dict, Optional, List
from pathlib import Path
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from PIL import Image
import io
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")
load_dotenv()
logger = logging.getLogger(__name__)

class LayoutSpecialistAgent:
    """
    Layout Specialist Agent for handling layout and abstract questions
    
    Uses hybrid approach:
    - Document AI OCR for text extraction and structure
    - Gemini models (2.5 Flash/Pro) with multimodal capabilities
    - Specialized prompting for layout reasoning
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize Layout Specialist Agent
        
        Args:
            model_name: Name of the Gemini model to use
        """
        self.model_name = model_name
        
        # Initialize Vertex AI
        project_id = os.getenv('GCP_PROJECT_ID')
        location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        
        if not project_id:
            raise ValueError("GCP_PROJECT_ID environment variable is required")
        
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_name)
        
        # Rate limiting
        self.last_request_time = None
        self.min_request_interval = 0.5
        
        logger.info(f"Layout Specialist Agent initialized with {model_name}")
    
    def analyze_layout_question(
        self, 
        image_path: str, 
        question: str, 
        question_types: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze layout/abstract question using hybrid OCR + Vision approach
        
        Args:
            image_path: Path to the document image
            question: Question to answer
            question_types: List of question types
            
        Returns:
            Dictionary with answer, confidence, and metadata
        """
        # Rate limiting
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
        
        try:
            # Extract OCR data for structural understanding
            ocr_data = self._extract_ocr_data(image_path)
            
            # Prepare image for vision analysis
            image_part = self._prepare_image(image_path)
            
            # Create specialized layout prompt
            prompt = self._create_layout_prompt(question, question_types, ocr_data)
            
            # Phase 1.5: Optimize temperature per question type
            temperature = 0.1  # Default for layout analysis
            if question_types:
                if 'layout' in question_types:
                    temperature = 0.1  # Balanced for structural analysis
                elif 'others' in question_types:
                    temperature = 0.15  # Allow more flexibility for abstract questions
            
            # Generate response
            start_time = time.time()
            response = self.model.generate_content(
                [prompt, image_part],
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": 400,  # Higher limit for complex layout descriptions
                    "top_p": 0.8,
                    "top_k": 40
                }
            )
            processing_time = time.time() - start_time
            self.last_request_time = time.time()
            
            # Extract answer
            try:
                answer = response.text.strip() if response.text else ""
            except ValueError as e:
                if "MAX_TOKENS" in str(e) or "finish_reason" in str(e):
                    logger.warning(f"Response truncated due to token limit: {e}")
                    answer = "Response truncated - complex layout analysis"
                else:
                    raise e
            
            # Calculate confidence based on response quality
            confidence = self._calculate_confidence(answer, ocr_data, question_types)
            
            logger.info(f"Layout Specialist: '{answer[:50]}...' (confidence: {confidence:.2f})")
            
            return {
                "answer": answer,
                "confidence": confidence,
                "processing_time": processing_time,
                "status": "success",
                "agent_type": "layout_specialist",
                "ocr_data_quality": ocr_data.get("quality", "unknown") if ocr_data.get("status") == "success" else "error",
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Layout analysis failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "confidence": 0.0,
                "processing_time": 0.0,
                "status": "error",
                "agent_type": "layout_specialist",
                "error": str(e)
            }
    
    def _extract_ocr_data(self, image_path: str) -> Dict:
        """
        Extract OCR data using Document AI for structural understanding
        
        Args:
            image_path: Path to the document image
            
        Returns:
            Dictionary with OCR data
        """
        try:
            from tools.document_ocr import process_document_with_ocr
            
            # Process document with Document AI
            ocr_result_json = process_document_with_ocr(image_path)
            ocr_result = json.loads(ocr_result_json)
            
            if ocr_result.get('status') == 'error':
                logger.warning(f"Document AI OCR failed: {ocr_result.get('error_message', 'Unknown error')}")
                return {"status": "error", "error": ocr_result.get('error_message', 'OCR failed')}
            
            # Extract structured data
            document = ocr_result.get('document', {})
            pages = document.get('pages', [])
            
            if not pages:
                return {"status": "error", "error": "No pages found in OCR result"}
            
            # Extract text and structure information
            full_text = document.get('text', '')
            page_count = len(pages)
            
            # Analyze text quality for layout questions
            text_length = len(full_text)
            word_count = len(full_text.split())
            
            # Determine quality based on text extraction
            if text_length > 500 and word_count > 50:
                quality = "excellent"
            elif text_length > 200 and word_count > 20:
                quality = "good"
            elif text_length > 50:
                quality = "fair"
            else:
                quality = "poor"
            
            return {
                "status": "success",
                "text": full_text,
                "text_length": text_length,
                "word_count": word_count,
                "page_count": page_count,
                "quality": quality,
                "pages": pages
            }
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _prepare_image(self, image_path: str) -> Part:
        """
        Prepare image for Gemini processing
        
        Args:
            image_path: Path to the document image
            
        Returns:
            Part object for Gemini
        """
        # Open image and resize if too large
        with Image.open(image_path) as img:
            # Resize if image is too large (reduce token usage)
            max_size = 1024  # Max dimension
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                logger.debug(f"Resized image from {img.size} to {new_size}")
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG', optimize=True)
            image_bytes = img_bytes.getvalue()
        
        return Part.from_data(image_bytes, mime_type='image/png')
    
    def _create_layout_prompt(self, question: str, question_types: Optional[List[str]], ocr_data: Dict) -> str:
        """
        Create specialized prompt for layout/abstract questions
        
        Args:
            question: The question to answer
            question_types: List of question types
            ocr_data: OCR extraction results
            
        Returns:
            Specialized prompt string
        """
        # Extract OCR text for context
        ocr_text = ocr_data.get('text', '')[:2000]  # Limit OCR text to avoid token limits
        ocr_quality = ocr_data.get('quality', 'unknown')
        
        # Build question type context
        type_context = ""
        if question_types:
            type_context = f"Question types: {', '.join(question_types)}"
        
        prompt = f"""You are a Layout Specialist Agent specializing in document layout analysis and spatial reasoning.

TASK: Answer the question by analyzing both the visual layout and extracted text.

QUESTION: {question}
{type_context}

OCR EXTRACTED TEXT (Quality: {ocr_quality}):
{ocr_text}

ANALYSIS INSTRUCTIONS:
1. **Visual Layout Analysis**: Examine the image for:
   - Document structure and organization
   - Spatial relationships between elements
   - Headers, sections, and formatting
   - Visual hierarchy and layout patterns

2. **Text Integration**: Combine OCR text with visual observations:
   - Cross-reference text with visual positioning
   - Identify layout-specific information not captured in OCR
   - Consider spatial context for text interpretation

3. **Layout-Specific Reasoning**: For layout questions, focus on:
   - Document structure and organization
   - Spatial relationships and positioning
   - Visual hierarchy and formatting
   - Abstract concepts that require spatial understanding

4. **Answer Format**: Provide ONLY the direct answer, nothing else. Be concise and precise.

RESPONSE: Answer the question by combining visual layout analysis with OCR text understanding. Provide only the direct answer."""

        return prompt
    
    def _calculate_confidence(self, answer: str, ocr_data: Dict, question_types: Optional[List[str]]) -> float:
        """
        Calculate confidence score for layout analysis
        
        Args:
            answer: Generated answer
            ocr_data: OCR extraction results
            question_types: List of question types
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not answer or answer.startswith("Error:") or answer.startswith("Response truncated"):
            return 0.0
        
        # Base confidence
        confidence = 0.7
        
        # Boost confidence based on OCR quality
        ocr_quality = ocr_data.get('quality', 'unknown')
        if ocr_quality == 'excellent':
            confidence += 0.15
        elif ocr_quality == 'good':
            confidence += 0.10
        elif ocr_quality == 'fair':
            confidence += 0.05
        
        # Boost confidence for layout-specific questions
        if question_types and 'layout' in question_types:
            confidence += 0.10
        
        # Boost confidence for longer, more detailed answers
        if len(answer) > 50:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def should_handle_question(self, question_types: Optional[List[str]]) -> bool:
        """
        Determine if this agent should handle the question
        
        Args:
            question_types: List of question types
            
        Returns:
            True if this agent should handle the question
        """
        if not question_types:
            return False
        
        # Layout specialist is best for these question types
        layout_specialist_types = {
            'layout',  # Primary specialty
            'others'   # Generic questions that benefit from layout analysis
        }
        
        return any(qt in layout_specialist_types for qt in question_types)
    
    def get_agent_info(self) -> Dict:
        """
        Get agent information
        
        Returns:
            Dictionary with agent metadata
        """
        return {
            "name": "Layout Specialist Agent",
            "description": "Specialized in layout and abstract questions using hybrid OCR + Vision",
            "model": self.model_name,
            "specialties": ["layout", "others"],
            "version": "1.0.0"
        }


def create_layout_specialist(model_name: Optional[str] = None) -> LayoutSpecialistAgent:
    """
    Create a Layout Specialist Agent instance
    
    Args:
        model_name: Optional model name (uses default from .env if None)
    
    Returns:
        LayoutSpecialistAgent instance
    """
    if model_name is None:
        model_name = os.getenv('MODEL', 'gemini-2.5-flash')
    return LayoutSpecialistAgent(model_name)

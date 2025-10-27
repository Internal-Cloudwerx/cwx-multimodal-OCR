"""
OCR Specialist Agent
Specialized agent for handling table/form questions using Document AI OCR + Gemini
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
from PIL import Image, ImageEnhance, ImageFilter
from dotenv import load_dotenv

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class OCRSpecialistAgent:
    """
    OCR Specialist Agent for table/form questions
    
    This agent is optimized for structured data extraction and table/form processing
    using Document AI OCR combined with Gemini. It focuses on questions that benefit
    from precise text extraction and structured data understanding.
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize OCR Specialist Agent
        
        Args:
            model_name: Gemini model to use (default: gemini-2.5-flash)
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
        self.min_request_interval = 0.5  # 0.5 seconds between requests
        
        logger.info(f"OCR Specialist Agent initialized with {model_name}")
    
    def analyze_table_form_question(
        self, 
        image_path: str, 
        question: str,
        question_types: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze a table/form question using Document AI OCR + Gemini
        
        Args:
            image_path: Path to the document image
            question: The question to answer
            question_types: List of question types (for context)
            
        Returns:
            Dictionary with answer, confidence, and metadata
        """
        try:
            # Rate limiting
            if self.last_request_time:
                elapsed = time.time() - self.last_request_time
                if elapsed < self.min_request_interval:
                    sleep_time = self.min_request_interval - elapsed
                    logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)
            
            # Step 1: Extract OCR data using Document AI
            ocr_data = self._extract_ocr_data(image_path)
            
            # Step 2: Load and optimize image
            image_part = self._prepare_image(image_path)
            
            # Step 3: Create specialized prompt for table/form questions
            prompt = self._create_ocr_prompt(question, question_types, ocr_data)
            
            # Step 4: Generate response
            start_time = time.time()
            response = self.model.generate_content(
                [prompt, image_part],
                generation_config={
                    "temperature": 0.05,  # Very low temperature for precise answers
                    "max_output_tokens": 400,  # Increased for complete answers
                    "top_p": 0.8,
                    "top_k": 40
                }
            )
            
            processing_time = time.time() - start_time
            self.last_request_time = time.time()
            
            # Step 5: Extract answer with better error handling
            try:
                answer = response.text.strip() if response.text else ""
            except ValueError as e:
                if "MAX_TOKENS" in str(e) or "finish_reason" in str(e):
                    logger.warning(f"Response truncated due to token limit: {e}")
                    answer = "Response truncated - complex table/form"
                else:
                    raise e
            
            # Step 6: Calculate confidence based on OCR data quality
            confidence = self._calculate_confidence(answer, ocr_data, response)
            
            result = {
                "answer": answer,
                "confidence": confidence,
                "processing_time": processing_time,
                "model_used": self.model_name,
                "agent_type": "ocr_specialist",
                "question_types": question_types or [],
                "ocr_data_quality": self._assess_ocr_quality(ocr_data),
                "status": "success"
            }
            
            logger.info(f"OCR Specialist: '{answer[:50]}...' (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"OCR Specialist failed: {e}", exc_info=True)
            return {
                "answer": "",
                "confidence": 0.0,
                "processing_time": 0.0,
                "model_used": self.model_name,
                "agent_type": "ocr_specialist",
                "question_types": question_types or [],
                "ocr_data_quality": "unknown",
                "status": "error",
                "error": str(e)
            }
    
    def _extract_ocr_data(self, image_path: str) -> Dict:
        """
        Extract OCR data using Document AI
        
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
            
            # Extract relevant data
            ocr_data = {
                "status": "success",
                "full_text": ocr_result.get('full_text', ''),
                "tables": ocr_result.get('tables', []),
                "entities": ocr_result.get('entities', []),
                "num_pages": ocr_result.get('num_pages', 0),
                "text_length": len(ocr_result.get('full_text', ''))
            }
            
            logger.debug(f"OCR extracted {ocr_data['text_length']} characters, {len(ocr_data['tables'])} tables")
            return ocr_data
            
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
            import io
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG', optimize=True)
            image_bytes = img_bytes.getvalue()
        
        return Part.from_data(image_bytes, mime_type='image/png')
    
    
    def _create_ocr_prompt(self, question: str, question_types: Optional[List[str]], ocr_data: Dict) -> str:
        """
        Create specialized prompt for OCR-based questions
        
        Args:
            question: The question to answer
            question_types: List of question types for context
            ocr_data: Extracted OCR data
            
        Returns:
            Formatted prompt string
        """
        # Determine question context
        context_hints = []
        if question_types:
            if 'table/list' in question_types:
                context_hints.append("This is a table/list question - focus on structured data")
            if 'form' in question_types:
                context_hints.append("This is a form question - look for form fields and values")
            if 'figure/diagram' in question_types:
                context_hints.append("This may contain charts/diagrams - look for numerical data")
        
        context_text = "\n".join(context_hints) if context_hints else ""
        
        # Build OCR context
        ocr_context = ""
        if ocr_data.get('status') == 'success':
            # Add full text (truncated)
            full_text = ocr_data.get('full_text', '')[:2000]  # First 2000 chars
            ocr_context += f"OCR Extracted Text:\n{full_text}\n"
            
            # Add table data if available
            tables = ocr_data.get('tables', [])
            if tables:
                ocr_context += "\nExtracted Tables:\n"
                for i, table in enumerate(tables[:3]):  # Max 3 tables
                    ocr_context += f"\nTable {i+1}:\n"
                    if table.get('headers'):
                        ocr_context += f"Headers: {', '.join(table['headers'])}\n"
                    if table.get('rows'):
                        for row in table['rows'][:10]:  # First 10 rows
                            ocr_context += f"  {', '.join(row)}\n"
            
            # Add entities if available
            entities = ocr_data.get('entities', [])
            if entities:
                ocr_context += "\nExtracted Entities:\n"
                for entity in entities[:10]:  # First 10 entities
                    ocr_context += f"  {entity.get('type', 'unknown')}: {entity.get('value', '')}\n"
        
        prompt = f"""You are an OCR Specialist Agent specialized in analyzing structured documents to answer questions.

Question: {question}

{context_text}

{ocr_context}

Instructions:
- Use BOTH the document image AND the OCR text above
- The OCR text provides precise text extraction, but the image shows layout and visual context
- For table questions, focus on the structured table data
- For form questions, look for form fields and their values
- For numerical questions, extract exact numbers from the OCR text
- Provide ONLY the direct answer, nothing else
- Be precise and accurate with numbers, dates, and names
- If the answer is a number, provide just that value
- If multiple values are found, provide the most relevant one

Answer:"""
        
        return prompt
    
    def _calculate_confidence(self, answer: str, ocr_data: Dict, response) -> float:
        """
        Calculate confidence score based on answer characteristics and OCR quality
        
        Args:
            answer: The generated answer
            ocr_data: OCR extraction data
            response: The full response object
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not answer:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence for specific answer patterns
        if answer.isdigit() or answer.replace('.', '').isdigit():
            confidence += 0.2  # Numeric answers are usually confident
        
        if len(answer) < 100:
            confidence += 0.1  # Concise answers are usually confident
        
        # Increase confidence based on OCR quality
        if ocr_data.get('status') == 'success':
            text_length = ocr_data.get('text_length', 0)
            if text_length > 100:
                confidence += 0.1  # Good OCR extraction
            
            tables = ocr_data.get('tables', [])
            if tables:
                confidence += 0.1  # Structured data available
        
        # Decrease confidence for uncertain language
        uncertain_words = ['maybe', 'possibly', 'might', 'could', 'seems', 'appears']
        if any(word in answer.lower() for word in uncertain_words):
            confidence -= 0.2
        
        # Ensure confidence is between 0.0 and 1.0
        return max(0.0, min(1.0, confidence))
    
    def _assess_ocr_quality(self, ocr_data: Dict) -> str:
        """
        Assess the quality of OCR extraction
        
        Args:
            ocr_data: OCR extraction data
            
        Returns:
            Quality assessment string
        """
        if ocr_data.get('status') != 'success':
            return "poor"
        
        text_length = ocr_data.get('text_length', 0)
        tables = ocr_data.get('tables', [])
        
        if text_length > 1000 and tables:
            return "excellent"
        elif text_length > 500:
            return "good"
        elif text_length > 100:
            return "fair"
        else:
            return "poor"
    
    def is_suitable_for_question(self, question_types: Optional[List[str]]) -> bool:
        """
        Determine if this agent is suitable for the given question types
        
        Args:
            question_types: List of question types
            
        Returns:
            True if this agent should handle the question
        """
        if not question_types:
            return False
        
        # OCR specialist is best for these question types
        ocr_specialist_types = {
            'table/list',
            'form',
            'free_text',
            'others'  # Generic questions that benefit from OCR
        }
        
        return any(qt in ocr_specialist_types for qt in question_types)
    
    def get_agent_info(self) -> Dict:
        """
        Get information about this agent
        
        Returns:
            Dictionary with agent metadata
        """
        return {
            "name": "OCR Specialist Agent",
            "description": "Specialized in table/form questions using Document AI OCR + Gemini",
            "model": self.model_name,
            "specialties": ["table/list", "form", "free_text", "others"],
            "version": "1.0.0"
        }


def create_ocr_specialist(model_name: Optional[str] = None) -> OCRSpecialistAgent:
    """
    Factory function to create an OCR Specialist Agent
    
    Args:
        model_name: Optional model name (uses default from .env if None)
    
    Returns:
        Initialized OCRSpecialistAgent instance
    """
    if model_name is None:
        model_name = os.getenv('MODEL', 'gemini-2.5-flash')
    return OCRSpecialistAgent(model_name)


if __name__ == "__main__":
    # Test the OCR Specialist Agent
    import tempfile
    from PIL import Image
    
    # Create a test image
    test_image = Image.new('RGB', (100, 100), color='white')
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        test_image.save(tmp_file.name, 'PNG')
        test_path = tmp_file.name
    
    try:
        # Initialize agent
        agent = create_ocr_specialist()
        
        # Test question
        result = agent.analyze_table_form_question(
            image_path=test_path,
            question="What is the total amount?",
            question_types=["table/list"]
        )
        
        print("OCR Specialist Test Result:")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Status: {result['status']}")
        
    finally:
        # Cleanup
        os.unlink(test_path)

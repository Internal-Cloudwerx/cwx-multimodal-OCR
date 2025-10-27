"""
Vision Specialist Agent
Specialized agent for handling image/photo questions using Gemini Vision models
"""

import os
import time
import logging
import warnings
from typing import Dict, Optional, List
from pathlib import Path
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from dotenv import load_dotenv

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class VisionSpecialistAgent:
    """
    Vision Specialist Agent for image/photo questions
    
    This agent is optimized for visual document understanding and question answering
    using Gemini Vision models. It focuses on image-based questions that require
    visual analysis rather than structured text extraction.
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize Vision Specialist Agent
        
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
        
        logger.info(f"Vision Specialist Agent initialized with {model_name}")
    
    def analyze_image_question(
        self, 
        image_path: str, 
        question: str,
        question_types: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze an image-based question using Gemini Vision
        
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
            
            # Load and optimize image
            from PIL import Image
            
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
            
            image_part = Part.from_data(image_bytes, mime_type='image/png')
            
            # Create specialized prompt for image questions
            prompt = self._create_vision_prompt(question, question_types)
            
            # Generate response
            start_time = time.time()
            response = self.model.generate_content(
                [prompt, image_part],
                generation_config={
                    "temperature": 0.1,  # Low temperature for consistent answers
                    "max_output_tokens": 200,  # Increased for better answers
                    "top_p": 0.8,
                    "top_k": 40
                }
            )
            
            processing_time = time.time() - start_time
            self.last_request_time = time.time()
            
            # Extract answer with better error handling
            try:
                answer = response.text.strip() if response.text else ""
            except ValueError as e:
                if "MAX_TOKENS" in str(e) or "finish_reason" in str(e):
                    logger.warning(f"Response truncated due to token limit: {e}")
                    answer = "Response truncated - image too large"
                else:
                    raise e
            
            # Calculate confidence based on response characteristics
            confidence = self._calculate_confidence(answer, response)
            
            result = {
                "answer": answer,
                "confidence": confidence,
                "processing_time": processing_time,
                "model_used": self.model_name,
                "agent_type": "vision_specialist",
                "question_types": question_types or [],
                "status": "success"
            }
            
            logger.info(f"Vision Specialist: '{answer[:50]}...' (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Vision Specialist failed: {e}", exc_info=True)
            return {
                "answer": "",
                "confidence": 0.0,
                "processing_time": 0.0,
                "model_used": self.model_name,
                "agent_type": "vision_specialist",
                "question_types": question_types or [],
                "status": "error",
                "error": str(e)
            }
    
    def _create_vision_prompt(self, question: str, question_types: Optional[List[str]] = None) -> str:
        """
        Create specialized prompt for vision-based questions
        
        Args:
            question: The question to answer
            question_types: List of question types for context
            
        Returns:
            Formatted prompt string
        """
        # Determine question context
        context_hints = []
        if question_types:
            if 'Image/Photo' in question_types:
                context_hints.append("This is an image/photo question - focus on visual elements")
            if 'handwritten' in question_types:
                context_hints.append("This may contain handwriting - look carefully at text")
            if 'Yes/No' in question_types:
                context_hints.append("This is a yes/no question - provide clear yes or no answer")
        
        context_text = "\n".join(context_hints) if context_hints else ""
        
        prompt = f"""You are a Vision Specialist Agent specialized in analyzing document images to answer questions.

Question: {question}

{context_text}

Instructions:
- Analyze the document image carefully
- Focus on visual elements, layout, and any visible text
- For image/photo questions, pay special attention to visual content
- For handwritten content, examine the handwriting carefully
- Provide ONLY the direct answer - no explanations or context
- Be precise and concise
- If the answer is a number, date, or name, provide just that value
- If unsure, provide your best estimate with appropriate confidence

Answer:"""
        
        return prompt
    
    def _calculate_confidence(self, answer: str, response) -> float:
        """
        Calculate confidence score based on answer characteristics
        
        Args:
            answer: The generated answer
            response: The full response object
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not answer:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence for specific answer patterns
        if answer.lower() in ['yes', 'no', 'true', 'false']:
            confidence += 0.2  # Binary answers are usually confident
        
        if answer.isdigit() or answer.replace('.', '').isdigit():
            confidence += 0.2  # Numeric answers are usually confident
        
        if len(answer) < 50:
            confidence += 0.1  # Concise answers are usually confident
        
        # Decrease confidence for uncertain language
        uncertain_words = ['maybe', 'possibly', 'might', 'could', 'seems', 'appears']
        if any(word in answer.lower() for word in uncertain_words):
            confidence -= 0.2
        
        # Ensure confidence is between 0.0 and 1.0
        return max(0.0, min(1.0, confidence))
    
    def is_suitable_for_question(self, question_types: Optional[List[str]] = None) -> bool:
        """
        Determine if this agent is suitable for the given question types
        
        Args:
            question_types: List of question types
            
        Returns:
            True if this agent should handle the question
        """
        if not question_types:
            return False
        
        # Vision specialist is best for these question types
        vision_specialist_types = {
            'Image/Photo',
            'handwritten',
            'Yes/No',  # Simple questions that benefit from visual analysis
            'figure/diagram'  # Charts and diagrams need visual analysis
        }
        
        return any(qt in vision_specialist_types for qt in question_types)
    
    def get_agent_info(self) -> Dict:
        """
        Get information about this agent
        
        Returns:
            Dictionary with agent metadata
        """
        return {
            "name": "Vision Specialist Agent",
            "description": "Specialized in image/photo questions using Gemini Vision models",
            "model": self.model_name,
            "specialties": ["Image/Photo", "handwritten", "Yes/No", "figure/diagram"],
            "version": "1.0.0"
        }


def create_vision_specialist(model_name: Optional[str] = None) -> VisionSpecialistAgent:
    """
    Factory function to create a Vision Specialist Agent
    
    Args:
        model_name: Optional model name (uses default from .env if None)
    
    Returns:
        Initialized VisionSpecialistAgent instance
    """
    if model_name is None:
        model_name = os.getenv('MODEL', 'gemini-2.5-flash')
    return VisionSpecialistAgent(model_name)


if __name__ == "__main__":
    # Test the Vision Specialist Agent
    import tempfile
    from PIL import Image
    
    # Create a test image
    test_image = Image.new('RGB', (100, 100), color='white')
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        test_image.save(tmp_file.name, 'PNG')
        test_path = tmp_file.name
    
    try:
        # Initialize agent
        agent = create_vision_specialist()
        
        # Test question
        result = agent.analyze_image_question(
            image_path=test_path,
            question="What color is this image?",
            question_types=["Image/Photo"]
        )
        
        print("Vision Specialist Test Result:")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Status: {result['status']}")
        
    finally:
        # Cleanup
        os.unlink(test_path)

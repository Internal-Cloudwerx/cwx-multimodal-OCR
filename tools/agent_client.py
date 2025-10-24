"""
ADK Agent REST API Client

Provides a client for communicating with ADK agents via REST API.
This avoids InMemoryRunner session management issues while enabling
proper agent orchestration for multi-agent systems.
"""

import requests
import logging
import time
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class AgentClient:
    """
    Client for communicating with ADK agent via REST API
    
    Usage:
        # Start agent server first: adk web --port 4200
        
        client = AgentClient(base_url="http://localhost:4200")
        response = client.ask(
            question="What is the total amount?",
            image_path="invoice.png"
        )
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:4200",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize agent client
        
        Args:
            base_url: Base URL of ADK agent server (from `adk web`)
            timeout: Request timeout in seconds
            max_retries: Number of retries for failed requests
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.session_id = None
        
    def _check_server(self) -> bool:
        """Check if agent server is running"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=2
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def ask(
        self,
        question: str,
        image_path: Optional[str] = None,
        context: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ask the agent a question
        
        Args:
            question: The question to ask
            image_path: Optional path to document image
            context: Optional additional context (e.g., OCR text)
            session_id: Optional session ID for conversation continuity
            
        Returns:
            Dict with keys:
                - answer: The agent's response
                - session_id: Session ID for follow-up questions
                - metadata: Additional response metadata
        """
        # Build message
        message_parts = []
        
        # Add context if provided
        if context:
            message_parts.append(f"Context:\n{context}\n")
        
        # Add image reference if provided
        if image_path:
            message_parts.append(f"Document: {image_path}\n")
        
        # Add question
        message_parts.append(f"Question: {question}")
        
        message = "\n".join(message_parts)
        
        # Prepare request
        payload = {
            "message": message,
            "session_id": session_id or self.session_id
        }
        
        # Add image as attachment if provided
        files = None
        if image_path and Path(image_path).exists():
            files = {
                'image': open(image_path, 'rb')
            }
        
        # Send request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    files=files,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Store session ID
                    if 'session_id' in result:
                        self.session_id = result['session_id']
                    
                    return {
                        'answer': result.get('response', result.get('answer', '')),
                        'session_id': result.get('session_id'),
                        'metadata': result.get('metadata', {})
                    }
                else:
                    logger.warning(
                        f"Agent API returned status {response.status_code}: "
                        f"{response.text}"
                    )
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(1 * (attempt + 1))  # Exponential backoff
                    
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Request to agent API failed (attempt {attempt + 1}): {e}"
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(1 * (attempt + 1))
            
            finally:
                if files:
                    files['image'].close()
        
        raise RuntimeError(
            f"Failed to get response from agent after {self.max_retries} attempts"
        )
    
    def reset_session(self):
        """Reset session (start fresh conversation)"""
        self.session_id = None


class AgentClientManager:
    """
    Manages agent client with automatic fallback
    
    Tries REST API first, falls back to direct tool calls if unavailable.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:4200",
        enable_fallback: bool = True
    ):
        """
        Initialize manager
        
        Args:
            base_url: Agent server URL
            enable_fallback: Whether to fall back to direct calls if API unavailable
        """
        self.base_url = base_url
        self.enable_fallback = enable_fallback
        self.client = AgentClient(base_url=base_url)
        self.api_available = self._check_api()
        
        if not self.api_available:
            if enable_fallback:
                logger.warning(
                    f"Agent API not available at {base_url}. "
                    f"Fallback to direct tool calls is enabled."
                )
            else:
                raise RuntimeError(
                    f"Agent API not available at {base_url}. "
                    f"Start agent server with: adk web --port 4200"
                )
    
    def _check_api(self) -> bool:
        """Check if agent API is available"""
        return self.client._check_server()
    
    def ask(
        self,
        question: str,
        image_path: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Ask question via agent API or fallback
        
        Args:
            question: The question
            image_path: Optional document image path
            context: Optional context (e.g., OCR text)
            
        Returns:
            Answer string
        """
        if self.api_available:
            try:
                result = self.client.ask(
                    question=question,
                    image_path=image_path,
                    context=context
                )
                return result['answer']
            except Exception as e:
                logger.error(f"Agent API call failed: {e}")
                
                if not self.enable_fallback:
                    raise
                
                logger.info("Falling back to direct tool calls")
                return self._fallback_ask(question, image_path, context)
        else:
            if not self.enable_fallback:
                raise RuntimeError("Agent API unavailable and fallback disabled")
            
            return self._fallback_ask(question, image_path, context)
    
    def _fallback_ask(
        self,
        question: str,
        image_path: Optional[str],
        context: Optional[str]
    ) -> str:
        """
        Fallback to direct tool calls
        
        This uses the same logic as the current docvqa_evaluator.py fallback.
        """
        from tools.document_ocr import process_document_with_ocr
        from vertexai.generative_models import GenerativeModel, Part
        import json
        import os
        
        # Initialize Gemini
        model = GenerativeModel(os.getenv('MODEL', 'gemini-2.5-flash'))
        
        # Get OCR if context not provided
        ocr_text = context
        if not ocr_text and image_path:
            try:
                ocr_result = json.loads(process_document_with_ocr(image_path))
                if ocr_result.get('status') != 'error':
                    ocr_text = ocr_result.get('full_text', '')[:1500]
            except Exception as e:
                logger.warning(f"OCR failed: {e}")
        
        # Build prompt
        if ocr_text:
            prompt = f"""You are analyzing a document. You have both the image and OCR text.

OCR Text:
{ocr_text}

Question: {question}

Provide ONLY the direct answer, nothing else."""
        else:
            prompt = f"""Analyze this document image.

Question: {question}

Provide ONLY the direct answer, nothing else."""
        
        # Generate response
        content_parts = [prompt]
        
        if image_path and Path(image_path).exists():
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            content_parts.append(Part.from_data(image_bytes, mime_type='image/png'))
        
        response = model.generate_content(
            content_parts,
            generation_config={
                'temperature': 0.05,
                'max_output_tokens': 75,
            }
        )
        
        return response.text.strip()


#!/usr/bin/env python3
"""
Answer Validator Agent

Validates and scores answers from specialist agents using multiple validation
techniques including confidence scoring, answer quality assessment, and
cross-validation with ground truth when available.
"""

import os
import time
import logging
import warnings
import re
import json
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")
load_dotenv()
logger = logging.getLogger(__name__)

class AnswerValidatorAgent:
    """
    Answer Validator Agent for validating and scoring specialist agent answers
    
    Provides:
    - Confidence scoring based on multiple factors
    - Answer quality assessment
    - Cross-validation with ground truth
    - Validation metadata and recommendations
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize Answer Validator Agent
        
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
        
        logger.info(f"Answer Validator Agent initialized with {model_name}")
    
    def validate_answer(
        self,
        answer: str,
        question: str,
        question_types: Optional[List[str]] = None,
        ground_truth: Optional[List[str]] = None,
        agent_result: Optional[Dict] = None,
        image_path: Optional[str] = None
    ) -> Dict:
        """
        Validate and score an answer from a specialist agent
        
        Args:
            answer: The answer to validate
            question: The original question
            question_types: List of question types
            ground_truth: List of correct answers (optional)
            agent_result: Full result from specialist agent (optional)
            image_path: Path to document image (optional)
            
        Returns:
            Dictionary with validation results, confidence scores, and metadata
        """
        # Rate limiting
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
        
        start_time = time.time()
        
        try:
            # Step 1: Basic answer validation
            basic_validation = self._validate_basic_answer(answer, question)
            
            # Step 2: Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                answer, question, question_types, agent_result
            )
            
            # Step 3: Cross-validate with ground truth if available
            ground_truth_validation = None
            if ground_truth:
                ground_truth_validation = self._validate_against_ground_truth(
                    answer, ground_truth, question_types
                )
            
            # Step 4: Advanced validation using Gemini (if needed)
            advanced_validation = None
            if self._needs_advanced_validation(answer, question_types):
                advanced_validation = self._advanced_validation(
                    answer, question, question_types, image_path
                )
            
            # Step 5: Compile final validation result
            processing_time = time.time() - start_time
            
            validation_result = {
                "answer": answer,
                "question": question,
                "validation_status": "success",
                "processing_time": processing_time,
                "agent_type": "answer_validator",
                "model": self.model_name,
                
                # Basic validation
                "basic_validation": basic_validation,
                
                # Confidence scores
                "confidence_scores": confidence_scores,
                "overall_confidence": confidence_scores["overall"],
                
                # Ground truth validation
                "ground_truth_validation": ground_truth_validation,
                
                # Advanced validation
                "advanced_validation": advanced_validation,
                
                # Metadata
                "question_types": question_types or [],
                "validation_timestamp": time.time()
            }
            
            logger.info(f"Answer Validator: Confidence {confidence_scores['overall']:.2f}, Status: {basic_validation['status']}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Answer validation failed: {e}")
            return {
                "answer": answer,
                "question": question,
                "validation_status": "error",
                "processing_time": time.time() - start_time,
                "agent_type": "answer_validator",
                "error": str(e),
                "confidence_scores": {"overall": 0.0},
                "basic_validation": {"status": "error", "issues": [str(e)]}
            }
    
    def _validate_basic_answer(self, answer: str, question: str) -> Dict:
        """
        Perform basic validation of the answer
        
        Args:
            answer: The answer to validate
            question: The original question
            
        Returns:
            Dictionary with basic validation results
        """
        issues = []
        warnings = []
        
        # Check for empty or very short answers
        if not answer or len(answer.strip()) == 0:
            issues.append("Empty answer")
        elif len(answer.strip()) < 2:
            warnings.append("Very short answer")
        
        # Check for error indicators
        error_indicators = ["error:", "failed:", "truncated", "cannot", "unable"]
        if any(indicator in answer.lower() for indicator in error_indicators):
            issues.append("Answer contains error indicators")
        
        # Check for uncertainty language
        uncertainty_words = ["maybe", "possibly", "might", "could", "seems", "appears", "likely"]
        uncertainty_count = sum(1 for word in uncertainty_words if word in answer.lower())
        if uncertainty_count > 0:
            warnings.append(f"Answer contains {uncertainty_count} uncertainty words")
        
        # Check answer length appropriateness
        if len(answer) > 500:
            warnings.append("Answer is very long")
        elif len(answer) > 1000:
            issues.append("Answer is excessively long")
        
        # Determine status
        if issues:
            status = "failed"
        elif warnings:
            status = "warning"
        else:
            status = "passed"
        
        return {
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "answer_length": len(answer),
            "has_uncertainty": uncertainty_count > 0
        }
    
    def _calculate_confidence_scores(
        self, 
        answer: str, 
        question: str, 
        question_types: Optional[List[str]], 
        agent_result: Optional[Dict]
    ) -> Dict:
        """
        Calculate multiple confidence scores for the answer
        
        Args:
            answer: The answer to score
            question: The original question
            question_types: List of question types
            agent_result: Full result from specialist agent
            
        Returns:
            Dictionary with various confidence scores
        """
        scores = {}
        
        # 1. Length-based confidence
        scores["length_confidence"] = self._calculate_length_confidence(answer, question)
        
        # 2. Format-based confidence
        scores["format_confidence"] = self._calculate_format_confidence(answer, question_types)
        
        # 3. Agent-based confidence
        scores["agent_confidence"] = agent_result.get("confidence", 0.5) if agent_result else 0.5
        
        # 4. Question type confidence
        scores["type_confidence"] = self._calculate_type_confidence(answer, question_types)
        
        # 5. Overall confidence (weighted average)
        weights = {
            "length_confidence": 0.2,
            "format_confidence": 0.3,
            "agent_confidence": 0.3,
            "type_confidence": 0.2
        }
        
        overall = sum(scores[key] * weights[key] for key in weights.keys())
        scores["overall"] = min(1.0, max(0.0, overall))
        
        return scores
    
    def _calculate_length_confidence(self, answer: str, question: str) -> float:
        """Calculate confidence based on answer length appropriateness"""
        if not answer:
            return 0.0
        
        answer_len = len(answer.strip())
        question_len = len(question.strip())
        
        # Ideal answer length is typically 10-200 characters
        if 10 <= answer_len <= 200:
            return 1.0
        elif 5 <= answer_len <= 500:
            return 0.8
        elif answer_len < 5:
            return 0.3
        else:
            return 0.6
    
    def _calculate_format_confidence(self, answer: str, question_types: Optional[List[str]]) -> float:
        """Calculate confidence based on answer format appropriateness"""
        if not answer or not question_types:
            return 0.5
        
        confidence = 0.5
        
        # Check for appropriate formats based on question type
        for q_type in question_types:
            if q_type in ['table/list', 'form']:
                # Should be concise, often numeric or short text
                if len(answer) < 100 and not any(word in answer.lower() for word in ['maybe', 'possibly']):
                    confidence += 0.2
            elif q_type in ['Yes/No']:
                # Should be yes/no or similar
                if answer.lower().strip() in ['yes', 'no', 'true', 'false', 'y', 'n']:
                    confidence += 0.3
            elif q_type in ['free_text']:
                # Can be longer, more descriptive
                if len(answer) > 20:
                    confidence += 0.1
        
        return min(1.0, confidence)
    
    def _calculate_type_confidence(self, answer: str, question_types: Optional[List[str]]) -> float:
        """Calculate confidence based on question type appropriateness"""
        if not question_types:
            return 0.5
        
        # Base confidence by question type (based on our performance data)
        type_confidence_map = {
            'table/list': 0.9,    # OCR specialist excels
            'form': 0.85,         # OCR specialist very good
            'layout': 0.8,        # Layout specialist good
            'free_text': 0.75,    # OCR specialist good
            'others': 0.7,       # OCR specialist decent
            'figure/diagram': 0.65, # Vision specialist improvement needed
            'handwritten': 0.6,   # Vision specialist improvement needed
            'Image/Photo': 0.6,   # Vision specialist improvement needed
            'Yes/No': 0.7        # Vision specialist decent
        }
        
        # Use the highest confidence for the question types present
        max_confidence = max(type_confidence_map.get(q_type, 0.5) for q_type in question_types)
        return max_confidence
    
    def _validate_against_ground_truth(
        self, 
        answer: str, 
        ground_truth: List[str], 
        question_types: Optional[List[str]]
    ) -> Dict:
        """
        Validate answer against ground truth
        
        Args:
            answer: The answer to validate
            ground_truth: List of correct answers
            question_types: List of question types
            
        Returns:
            Dictionary with ground truth validation results
        """
        if not ground_truth:
            return {"status": "no_ground_truth", "match": False, "similarity": 0.0}
        
        # Simple exact match
        answer_clean = answer.strip().lower()
        exact_match = any(gt.strip().lower() == answer_clean for gt in ground_truth)
        
        # Partial match (answer contains ground truth or vice versa)
        partial_match = any(
            gt.strip().lower() in answer_clean or answer_clean in gt.strip().lower()
            for gt in ground_truth
        )
        
        # Calculate similarity score
        similarity_scores = []
        for gt in ground_truth:
            gt_clean = gt.strip().lower()
            if answer_clean == gt_clean:
                similarity_scores.append(1.0)
            elif gt_clean in answer_clean or answer_clean in gt_clean:
                similarity_scores.append(0.8)
            else:
                # Simple character overlap
                overlap = len(set(answer_clean) & set(gt_clean))
                total = len(set(answer_clean) | set(gt_clean))
                similarity_scores.append(overlap / total if total > 0 else 0.0)
        
        max_similarity = max(similarity_scores) if similarity_scores else 0.0
        
        return {
            "status": "validated",
            "exact_match": exact_match,
            "partial_match": partial_match,
            "match": exact_match or partial_match,
            "similarity": max_similarity,
            "ground_truth_count": len(ground_truth),
            "best_match": ground_truth[similarity_scores.index(max_similarity)] if similarity_scores else None
        }
    
    def _needs_advanced_validation(self, answer: str, question_types: Optional[List[str]]) -> bool:
        """Determine if advanced validation is needed"""
        if not answer or len(answer.strip()) < 3:
            return False
        
        # Advanced validation for complex questions
        if question_types:
            complex_types = ['free_text', 'layout', 'others']
            if any(q_type in complex_types for q_type in question_types):
                return True
        
        # Advanced validation for long answers
        if len(answer) > 200:
            return True
        
        return False
    
    def _advanced_validation(
        self, 
        answer: str, 
        question: str, 
        question_types: Optional[List[str]], 
        image_path: Optional[str]
    ) -> Dict:
        """
        Perform advanced validation using Gemini
        
        Args:
            answer: The answer to validate
            question: The original question
            question_types: List of question types
            image_path: Path to document image
            
        Returns:
            Dictionary with advanced validation results
        """
        try:
            # Create validation prompt
            prompt = self._create_validation_prompt(answer, question, question_types)
            
            # Generate validation response
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 200,
                    "top_p": 0.8,
                    "top_k": 40
                }
            )
            
            validation_text = response.text.strip() if response.text else ""
            
            # Parse validation response
            return self._parse_validation_response(validation_text)
            
        except Exception as e:
            logger.warning(f"Advanced validation failed: {e}")
            return {
                "status": "failed",
                "reason": str(e),
                "confidence": 0.5
            }
    
    def _create_validation_prompt(self, answer: str, question: str, question_types: Optional[List[str]]) -> str:
        """Create prompt for advanced validation"""
        type_context = f"Question types: {', '.join(question_types)}" if question_types else "No specific question types"
        
        prompt = f"""You are an Answer Validator Agent. Evaluate the quality and correctness of this answer.

QUESTION: {question}
ANSWER: {answer}
{type_context}

Evaluate the answer on these criteria:
1. **Relevance**: Does the answer address the question?
2. **Completeness**: Is the answer complete and informative?
3. **Accuracy**: Does the answer appear factually correct?
4. **Clarity**: Is the answer clear and well-formatted?

Provide your assessment in this format:
RELEVANCE: [score 0-1]
COMPLETENESS: [score 0-1] 
ACCURACY: [score 0-1]
CLARITY: [score 0-1]
OVERALL: [score 0-1]
ISSUES: [list any issues or concerns]
RECOMMENDATIONS: [suggestions for improvement]"""

        return prompt
    
    def _parse_validation_response(self, validation_text: str) -> Dict:
        """Parse the validation response from Gemini"""
        try:
            # Extract scores using regex
            scores = {}
            for metric in ['RELEVANCE', 'COMPLETENESS', 'ACCURACY', 'CLARITY', 'OVERALL']:
                pattern = f"{metric}:\\s*([0-9.]+)"
                match = re.search(pattern, validation_text)
                if match:
                    scores[metric.lower()] = float(match.group(1))
                else:
                    scores[metric.lower()] = 0.5
            
            # Extract issues and recommendations
            issues = []
            recommendations = []
            
            if "ISSUES:" in validation_text:
                issues_section = validation_text.split("ISSUES:")[1].split("RECOMMENDATIONS:")[0].strip()
                issues = [issue.strip() for issue in issues_section.split('\n') if issue.strip()]
            
            if "RECOMMENDATIONS:" in validation_text:
                rec_section = validation_text.split("RECOMMENDATIONS:")[1].strip()
                recommendations = [rec.strip() for rec in rec_section.split('\n') if rec.strip()]
            
            return {
                "status": "completed",
                "scores": scores,
                "issues": issues,
                "recommendations": recommendations,
                "confidence": scores.get('overall', 0.5)
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse validation response: {e}")
            return {
                "status": "parse_error",
                "reason": str(e),
                "confidence": 0.5
            }
    
    def get_validator_info(self) -> Dict:
        """
        Get validator information
        
        Returns:
            Dictionary with validator metadata
        """
        return {
            "name": "Answer Validator Agent",
            "description": "Validates and scores answers from specialist agents",
            "model": self.model_name,
            "capabilities": [
                "Basic answer validation",
                "Multi-factor confidence scoring", 
                "Ground truth validation",
                "Advanced Gemini-based validation",
                "Answer quality assessment"
            ],
            "version": "1.0.0"
        }


def create_answer_validator() -> AnswerValidatorAgent:
    """
    Create an Answer Validator Agent instance
    
    Returns:
        AnswerValidatorAgent instance
    """
    return AnswerValidatorAgent()


if __name__ == "__main__":
    # Test the Answer Validator Agent
    validator = create_answer_validator()
    
    # Test cases
    test_cases = [
        {
            "answer": "Paul",
            "question": "To whom is the document sent?",
            "question_types": ["form"],
            "ground_truth": ["Paul"]
        },
        {
            "answer": "Response truncated - complex table/form",
            "question": "What is the total amount?",
            "question_types": ["table/list"],
            "ground_truth": ["$975.00"]
        },
        {
            "answer": "The name of the company is ITC Limited",
            "question": "What is the company name?",
            "question_types": ["layout"],
            "ground_truth": ["ITC Limited", "itc limited"]
        }
    ]
    
    print("Answer Validator Agent Test Results:")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases):
        result = validator.validate_answer(**test_case)
        
        print(f"\nTest {i+1}:")
        print(f"Question: {test_case['question']}")
        print(f"Answer: {test_case['answer']}")
        print(f"Overall Confidence: {result['confidence_scores']['overall']:.2f}")
        print(f"Basic Validation: {result['basic_validation']['status']}")
        
        if result['ground_truth_validation']:
            gt_val = result['ground_truth_validation']
            print(f"Ground Truth Match: {gt_val['match']} (similarity: {gt_val['similarity']:.2f})")
        
        print(f"Processing Time: {result['processing_time']:.2f}s")

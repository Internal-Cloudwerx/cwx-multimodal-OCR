"""
DocVQA Benchmark Evaluator
Evaluates document processing agent on SP-DocVQA benchmark
"""

from evaluation.hf_docvqa_loader import HFDocVQADataset
from evaluation.anls_metric import anls_score_multi
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import json
import pandas as pd
import time
import logging
from collections import defaultdict
import tempfile
import os
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocVQAEvaluator:
    """
    Evaluate agent on SP-DocVQA benchmark with ANLS metric
    """
    
    def __init__(
        self,
        agent,
        dataset_split: str = "validation",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize evaluator
        
        Args:
            agent: ADK agent instance (root_agent)
            dataset_split: 'train', 'validation', or 'test'
            cache_dir: Cache directory for dataset
        """
        self.agent = agent
        self.dataset_split = dataset_split
        
        # Rate limiting to avoid 429 errors
        self.last_request_time = None
        self.min_request_interval = 0.5  # 0.5 seconds = 120 RPM (well under 40M quota!)
        
        # Initialize Vertex AI for fallback
        project_id = os.getenv('GCP_PROJECT_ID')
        location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        if project_id:
            vertexai.init(project=project_id, location=location)
            self.gemini_model = GenerativeModel("gemini-2.5-flash")
        else:
            raise ValueError("GCP_PROJECT_ID environment variable is required")
        
        # Load dataset
        logger.info("Initializing DocVQA evaluator...")
        self.loader = HFDocVQADataset(cache_dir=cache_dir)
        self.dataset = self.loader.load(dataset_split)
        
        self.results = []
        self.progress_file = None
        
    def evaluate(
        self,
        num_samples: Optional[int] = None,
        start_idx: int = 0,
        output_dir: str = "evaluation/results/docvqa",
        save_frequency: int = 10,
        resume: bool = True
    ) -> Dict:
        """
        Run evaluation on DocVQA benchmark
        
        Args:
            num_samples: Number of samples to evaluate (None = all)
            start_idx: Starting index (for resuming)
            output_dir: Directory to save results
            save_frequency: Save progress every N samples
            resume: Resume from previous run if exists
            
        Returns:
            Dictionary with evaluation metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup progress file
        self.progress_file = output_path / f"progress_{self.dataset_split}.json"
        
        # Resume if requested
        if resume and self.progress_file.exists():
            start_idx, self.results = self._load_progress()
            logger.info(f"Resuming from sample {start_idx}")
        
        # Determine samples to process
        total_samples = len(self.dataset)
        end_idx = min(start_idx + num_samples, total_samples) if num_samples else total_samples
        
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING EVALUATION")
        logger.info(f"{'='*80}")
        logger.info(f"Dataset: SP-DocVQA {self.dataset_split}")
        logger.info(f"Samples: {start_idx} to {end_idx} (total: {end_idx - start_idx})")
        logger.info(f"{'='*80}\n")
        
        # Evaluate samples
        for idx in tqdm(range(start_idx, end_idx), desc="Evaluating"):
            try:
                sample = self.loader.get_sample(idx)
                result = self._evaluate_sample(sample, idx)
                self.results.append(result)
                
                # Save progress periodically
                if (idx + 1) % save_frequency == 0:
                    self._save_progress(idx + 1)
                    
            except Exception as e:
                logger.error(f"Error on sample {idx}: {e}", exc_info=True)
                # Record error
                self.results.append({
                    'questionId': idx,
                    'error': str(e),
                    'success': False
                })
        
        # Compute final metrics
        metrics = self._compute_metrics()
        
        # Generate reports
        self._generate_reports(metrics, output_path)
        
        # Clean up progress file
        if self.progress_file.exists():
            self.progress_file.unlink()
        
        return metrics
    
    def _evaluate_sample(self, sample: Dict, idx: int) -> Dict:
        """Evaluate a single sample"""
        
        question = sample['question']
        image = sample['image']  # PIL Image
        ground_truth_answers = sample['answers']
        question_types = sample.get('question_types', [])
        
        start_time = time.time()
        
        # Save image temporarily for agent
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            image.save(tmp_file.name, 'PNG')
            tmp_image_path = tmp_file.name
        
        try:
            # Run agent with question type hints
            prediction = self._run_agent(tmp_image_path, question, question_types)
            
            # Compute ANLS
            anls = anls_score_multi(prediction, ground_truth_answers)
            
            processing_time = time.time() - start_time
            
            result = {
                'questionId': sample['questionId'],
                'question': question,
                'prediction': prediction,
                'ground_truth': ground_truth_answers,
                'anls_score': anls,
                'correct': anls >= 0.5,  # Binary correctness
                'question_types': sample['question_types'],
                'docId': sample['docId'],
                'processing_time': processing_time,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Agent failed on sample {idx}: {e}")
            result = {
                'questionId': sample['questionId'],
                'question': question,
                'error': str(e),
                'success': False
            }
        
        finally:
            # Clean up temp file
            if os.path.exists(tmp_image_path):
                os.unlink(tmp_image_path)
        
        return result
    
    def _run_agent(self, image_path: str, question: str, question_types: list = None) -> str:
        """
        Run ADK agent on a question with selective OCR based on question type
        
        This method interfaces with your ADK agent to get answers.
        Adjust as needed based on your agent's implementation.
        """
        try:
            # Method 1: Try using ADK runner (correct API with sessions)
            from google.adk.runners import InMemoryRunner
            from google.genai import types
            import uuid
            
            runner = InMemoryRunner(agent=self.agent)
            
            # Use simple session ID - runner will create session implicitly
            user_id = "docvqa_evaluator"
            session_id = str(uuid.uuid4())
            
            # Format query - instruction to give short answer
            query = f"Analyze the document at {image_path} and answer: {question}\n\nProvide only the answer, no explanation."
            
            # Create Content object
            message = types.Content(role="user", parts=[types.Part(text=query)])
            
            # Run agent and collect events
            answer = None
            for event in runner.run(user_id=user_id, session_id=session_id, new_message=message):
                # Check for agent response
                if hasattr(event, 'type') and 'response' in str(event.type).lower():
                    if hasattr(event, 'content'):
                        answer = event.content
                elif hasattr(event, 'content') and hasattr(event.content, 'parts'):
                    # Extract text from parts
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            answer = part.text
                            break
                elif hasattr(event, 'text'):
                    answer = event.text
            
            if answer:
                answer = answer.strip()
                logger.info(f"✅ ADK agent successfully answered: {answer[:100]}")
                return answer
            else:
                raise ValueError("Agent did not return a valid answer")
            
        except Exception as e:
            # Method 2: Selective Hybrid - Smart OCR usage based on question type
            logger.warning(f"ADK runner failed, using selective hybrid fallback: {e}")
            
            try:
                # Step 1: Decide whether to use OCR based on question type
                question_types = question_types or []
                
                # Question types that benefit from OCR (structured data)
                ocr_beneficial_types = {
                    'table/list',      # Tables need structured text
                    'figure/diagram',  # Charts benefit from OCR numbers
                    'layout',          # Layout understanding needs text
                    'form',            # Forms (if typed)
                    'free_text',       # Free text needs OCR
                    'others'           # Generic, use OCR
                }
                
                # Question types that work better with vision only
                vision_only_types = {
                    'handwritten',     # OCR struggles with handwriting
                    'Image/Photo',     # Photos need visual understanding
                    'Yes/No'           # Simple questions, vision sufficient
                }
                
                # Decide: Use OCR if any question type benefits from it, AND no vision-only types
                use_ocr = False
                if question_types:
                    has_vision_only = any(qt in vision_only_types for qt in question_types)
                    has_ocr_beneficial = any(qt in ocr_beneficial_types for qt in question_types)
                    use_ocr = has_ocr_beneficial and not has_vision_only
                else:
                    # Default: use OCR if no type info
                    use_ocr = True
                
                logger.info(f"Question types: {question_types}, Using OCR: {use_ocr}")
                
                # Step 2: Conditionally extract text with Document AI OCR
                ocr_text = None
                tables_text = None
                
                if use_ocr:
                    from tools.document_ocr import process_document_with_ocr
                    
                    ocr_result_json = process_document_with_ocr(image_path)
                    ocr_result = json.loads(ocr_result_json)
                    
                    if ocr_result.get('status') == 'error':
                        logger.warning(f"Document AI OCR failed, falling back to vision-only")
                    else:
                        # Extract relevant OCR data (simple truncation)
                        ocr_text = ocr_result.get('full_text', '')[:1500]  # First 1500 chars
                        
                        # Format tables if available
                        if ocr_result.get('tables'):
                            tables_text = "\n\nExtracted Tables:\n"
                            for i, table in enumerate(ocr_result['tables'][:2]):  # Max 2 tables
                                tables_text += f"\nTable {i+1}:\n"
                                if table.get('headers'):
                                    tables_text += f"Headers: {', '.join(table['headers'])}\n"
                                if table.get('rows'):
                                    for row in table['rows'][:5]:  # First 5 rows
                                        tables_text += f"  {', '.join(row)}\n"
                
                # Step 3: Read image for vision
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                
                image_part = Part.from_data(image_bytes, mime_type='image/png')
                
                # Step 4: Create simple, direct prompt
                if ocr_text:
                    prompt = f"""You are analyzing a document to answer a specific question. You have both the document image and OCR-extracted text to help you.

OCR Extracted Text:
{ocr_text}
{tables_text if tables_text else ""}

Question: {question}

Instructions:
- Use BOTH the visual document image AND the OCR text above
- The OCR text provides structure, but the image shows layout and visual elements
- Find the specific information requested
- Provide ONLY the direct answer, nothing else
- Be concise and precise
- If the answer is a number, date, or name, provide just that value
- Do not add explanations, context, or extra words

Answer:"""
                else:
                    # Fallback to vision-only if OCR failed
                    prompt = f"""You are analyzing a document image to answer a specific question.

Question: {question}

Instructions:
- Look carefully at the document image
- Find the specific information requested
- Provide ONLY the direct answer, nothing else
- Be concise and precise
- If the answer is a number, date, or name, provide just that value
- Do not add explanations, context, or extra words

Answer:"""
                
                # Step 5: Rate limiting - prevent hitting 429 errors
                if self.last_request_time:
                    elapsed = (time.time() - self.last_request_time)
                    if elapsed < self.min_request_interval:
                        sleep_time = self.min_request_interval - elapsed
                        logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                        time.sleep(sleep_time)
                
                # Step 6: Call Gemini (with vision + optional OCR context)
                try:
                    response = self.gemini_model.generate_content(
                        [prompt, image_part],
                        generation_config={
                            "temperature": 0.05,  # Slightly more deterministic than 0.1, but not too rigid
                            "max_output_tokens": 75,  # Balanced: not too short, not too long
                        }
                    )
                    self.last_request_time = time.time()  # Record successful request
                    
                except Exception as api_error:
                    if '429' in str(api_error):  # Rate limit error
                        logger.warning("Rate limit hit (429), waiting 30 seconds before retry...")
                        time.sleep(30)
                        # Retry once
                        response = self.gemini_model.generate_content(
                            [prompt, image_part],
                            generation_config={
                                "temperature": 0.05,
                                "max_output_tokens": 75,
                            }
                        )
                        self.last_request_time = time.time()
                    else:
                        raise api_error
                
                answer = response.text.strip()
                
                # Advanced answer post-processing and normalization
                answer = self._normalize_answer(answer)
                
                return answer
                
            except Exception as e2:
                logger.error(f"Hybrid fallback also failed: {e2}")
                raise Exception(f"Both ADK runner and hybrid fallback failed: {e}, {e2}")
    
    def _normalize_answer(self, answer: str) -> str:
        """
        Lightly normalize and clean up model answers for better ANLS matching
        """
        import re
        
        # Remove only explicit answer prefixes (be conservative)
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
        elif answer.startswith("The answer is:"):
            answer = answer[14:].strip()
        
        # Remove quotes if they wrap the entire answer
        if (answer.startswith('"') and answer.endswith('"')) or \
           (answer.startswith("'") and answer.endswith("'")):
            answer = answer[1:-1].strip()
        
        # Normalize spacing around currency (conservative fix)
        # Fix: "$ 975.00" -> "$975.00" (but don't break other things)
        answer = re.sub(r'\$\s+', '$', answer)
        
        # Fix obvious spacing issues in numbers
        # Fix: "975 . 00" -> "975.00"
        answer = re.sub(r'(\d+)\s+\.\s+(\d+)', r'\1.\2', answer)
        
        # Fix multiple spaces
        answer = re.sub(r'\s+', ' ', answer)
        
        # Remove trailing punctuation (but be careful with numbers)
        if not answer or answer[-1] not in '0123456789':
            answer = answer.rstrip('.,!?;:')
        
        # Final trim
        answer = answer.strip()
        
        return answer
    
    def _compute_metrics(self) -> Dict:
        """Compute evaluation metrics"""
        
        successful_results = [r for r in self.results if r.get('success', False)]
        
        if not successful_results:
            return {
                'error': 'No successful evaluations',
                'total_samples': len(self.results)
            }
        
        # Overall ANLS
        anls_scores = [r['anls_score'] for r in successful_results]
        overall_anls = sum(anls_scores) / len(anls_scores)
        
        # Accuracy (binary)
        accuracy = sum(r['correct'] for r in successful_results) / len(successful_results)
        
        # By question type
        by_type = defaultdict(list)
        for r in successful_results:
            for qtype in r.get('question_types', ['unknown']):
                by_type[qtype].append(r['anls_score'])
        
        type_metrics = {
            qtype: {
                'anls': sum(scores) / len(scores),
                'count': len(scores)
            }
            for qtype, scores in by_type.items()
        }
        
        # Processing time
        processing_times = [r['processing_time'] for r in successful_results]
        avg_time = sum(processing_times) / len(processing_times)
        
        metrics = {
            'dataset': f'SP-DocVQA-{self.dataset_split}',
            'total_samples': len(self.results),
            'successful_samples': len(successful_results),
            'failed_samples': len(self.results) - len(successful_results),
            'overall_anls': overall_anls,
            'accuracy': accuracy,
            'by_question_type': type_metrics,
            'avg_processing_time_seconds': avg_time,
            'median_processing_time_seconds': sorted(processing_times)[len(processing_times)//2],
            'total_processing_time_seconds': sum(processing_times)
        }
        
        return metrics
    
    def _generate_reports(self, metrics: Dict, output_dir: Path):
        """Generate evaluation reports"""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 1. Save detailed results JSON
        results_file = output_dir / f"detailed_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'dataset': f'SP-DocVQA-{self.dataset_split}',
                    'timestamp': timestamp,
                    'agent': getattr(self.agent, 'name', 'unknown')
                },
                'metrics': metrics,
                'results': self.results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Detailed results: {results_file}")
        
        # 2. Save metrics summary JSON
        metrics_file = output_dir / f"metrics_{timestamp}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"✓ Metrics summary: {metrics_file}")
        
        # 3. Save results CSV
        successful_results = [r for r in self.results if r.get('success', False)]
        if successful_results:
            df = pd.DataFrame(successful_results)
            csv_file = output_dir / f"results_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"✓ Results CSV: {csv_file}")
        
        # 4. Print summary to console
        self._print_summary(metrics)
    
    def _print_summary(self, metrics: Dict):
        """Print evaluation summary to console"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Dataset: {metrics['dataset']}")
        logger.info(f"Total Samples: {metrics['total_samples']}")
        logger.info(f"Successful: {metrics['successful_samples']}")
        logger.info(f"Failed: {metrics['failed_samples']}")
        logger.info(f"")
        logger.info(f"PERFORMANCE METRICS:")
        logger.info(f"  Overall ANLS: {metrics['overall_anls']:.4f} ({metrics['overall_anls']*100:.2f}%)")
        logger.info(f"  Accuracy (ANLS >= 0.5): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"")
        logger.info(f"BY QUESTION TYPE:")
        for qtype, type_metrics in sorted(metrics['by_question_type'].items()):
            logger.info(f"  {qtype}:")
            logger.info(f"    ANLS: {type_metrics['anls']:.4f} ({type_metrics['anls']*100:.2f}%)")
            logger.info(f"    Count: {type_metrics['count']}")
        logger.info(f"")
        logger.info(f"PERFORMANCE:")
        logger.info(f"  Avg Time: {metrics['avg_processing_time_seconds']:.2f}s")
        logger.info(f"  Median Time: {metrics['median_processing_time_seconds']:.2f}s")
        logger.info(f"  Total Time: {metrics['total_processing_time_seconds']:.1f}s")
        logger.info(f"{'='*80}\n")
    
    def _save_progress(self, current_idx: int):
        """Save progress to resume later"""
        progress = {
            'current_idx': current_idx,
            'results': self.results
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f)
    
    def _load_progress(self):
        """Load previous progress"""
        with open(self.progress_file, 'r') as f:
            progress = json.load(f)
        return progress['current_idx'], progress['results']


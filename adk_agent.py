"""
ADK Multi-Agent Document Analysis Agent
This agent uses the multi-agent system for intelligent document analysis
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
from google.adk.agents import Agent
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Import our multi-agent system
from adk_a2a_wrapper import create_adk_a2a_wrapper

# Initialize the multi-agent wrapper
multi_agent_wrapper = create_adk_a2a_wrapper()

def extract_page_number_from_question(question: str) -> int:
    """
    Extract page number from question text.
    Returns 1 if no page number is specified.
    """
    import re
    
    # Look for patterns like "page 4", "page four", "4th page", etc.
    patterns = [
        r'page\s+(\d+)',
        r'page\s+(one|two|three|four|five|six|seven|eight|nine|ten)',
        r'(\d+)(?:st|nd|rd|th)\s+page',
        r'(\d+)\s+page'
    ]
    
    word_to_num = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    for pattern in patterns:
        match = re.search(pattern, question.lower())
        if match:
            if pattern.endswith('(one|two|three|four|five|six|seven|eight|nine|ten)'):
                return word_to_num.get(match.group(1), 1)
            else:
                return int(match.group(1))
    
    return 1  # Default to page 1

def detect_page_ambiguity(question: str, pdf_path: str) -> dict:
    """
    Detect if there's ambiguity between PDF page number and printed page number.
    Returns a dict with ambiguity info and clarification options.
    """
    import re
    
    # Check if question mentions a specific page number
    page_patterns = [
        r'page\s+(\d+)',
        r'(\d+)(?:st|nd|rd|th)\s+page',
        r'(\d+)\s+page'
    ]
    
    mentioned_page = None
    for pattern in page_patterns:
        match = re.search(pattern, question.lower())
        if match:
            mentioned_page = int(match.group(1))
            break
    
    if not mentioned_page:
        return {"has_ambiguity": False}
    
    # Try to detect printed page numbers in the PDF
    try:
        import fitz
        doc = fitz.open(pdf_path)
        
        # Check first few pages for printed page numbers
        printed_pages_found = []
        for page_num in range(min(5, len(doc))):  # Check first 5 pages
            page = doc[page_num]
            text = page.get_text()
            
            # Look for page numbers in common formats
            page_number_patterns = [
                r'\b(\d+)\b',  # Any number
                r'Page\s+(\d+)',
                r'page\s+(\d+)',
                r'-\s*(\d+)\s*-',  # - 5 -
                r'(\d+)\s*$',  # Number at end of line
            ]
            
            for pattern in page_number_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        num = int(match)
                        if 1 <= num <= len(doc):  # Reasonable page number range
                            printed_pages_found.append((page_num + 1, num))  # (PDF_page, printed_page)
                    except:
                        continue
        
        total_pages = len(doc)
        doc.close()
        
        # Check if mentioned page could be either PDF page or printed page
        pdf_page_match = mentioned_page <= total_pages
        printed_page_match = any(printed_page == mentioned_page for _, printed_page in printed_pages_found)
        
        if pdf_page_match and printed_page_match:
            return {
                "has_ambiguity": True,
                "mentioned_page": mentioned_page,
                "pdf_total_pages": total_pages,
                "printed_pages_found": printed_pages_found[:10],  # First 10 matches
                "clarification_needed": True
            }
        else:
            return {"has_ambiguity": False}
            
    except Exception as e:
        return {"has_ambiguity": False}

def convert_pdf_to_image(pdf_path: str, page_number: int = 1) -> str:
    """
    Convert a PDF page to an image file for processing.
    """
    try:
        import fitz  # PyMuPDF
        import tempfile
        from pathlib import Path
        
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        
        # Get the specified page (default to page 1, 0-indexed)
        page_index = page_number - 1
        if page_index >= len(pdf_document):
            page_index = 0  # Default to first page if page number is too high
        
        page = pdf_document[page_index]
        
        # Convert page to image
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        
        # Create temporary image file
        temp_dir = Path(tempfile.gettempdir())
        temp_image_path = temp_dir / f"pdf_page_{page_number}.png"
        
        # Save the image
        pix.save(str(temp_image_path))
        pdf_document.close()
        
        return str(temp_image_path)
        
    except ImportError:
        return pdf_path
    except Exception as e:
        return pdf_path

def find_recent_uploaded_files() -> list:
    """
    Find recently uploaded files that might be from ADK.
    """
    import os
    import time
    from pathlib import Path
    
    recent_files = []
    current_time = time.time()
    
    # Check common upload directories (including ADK-specific ones)
    upload_dirs = [
        "/tmp",
        "/var/tmp", 
        "/tmp/uploads",
        "/var/tmp/uploads",
        "/tmp/adk_uploads",  # ADK-specific
        "/var/tmp/adk_uploads",  # ADK-specific
        Path.cwd(),
        Path(__file__).parent / "uploads",
        Path(__file__).parent / "temp",
        Path.home() / "tmp",  # User temp directory
        Path.home() / "Downloads",  # Downloads directory
    ]
    
    for upload_dir in upload_dirs:
        if os.path.exists(upload_dir):
            try:
                for file in os.listdir(upload_dir):
                    file_path = os.path.join(upload_dir, file)
                    if os.path.isfile(file_path):
                        # Check if file was modified in the last 30 minutes (longer window for ADK)
                        file_time = os.path.getmtime(file_path)
                        if current_time - file_time < 1800:  # 30 minutes
                            recent_files.append(file_path)
            except:
                continue
    
    return recent_files

def resolve_file_path(image_path: str, question: str = "") -> str:
    """
    Resolve file path for different ADK scenarios.
    Handles relative paths, absolute paths, and common ADK file patterns.
    """
    import os
    from pathlib import Path
    
    # Special case: if user mentions "sample document" or similar, use the sample file
    if "sample" in question.lower() or "sample" in image_path.lower():
        sample_file = Path(__file__).parent / "data" / "sample" / "1800HistoryTestOCR.pdf"
        if sample_file.exists():
            return str(sample_file)
    
    # If it's already an absolute path and exists, use it
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    
    # Try relative to current directory
    current_dir = Path.cwd()
    relative_path = current_dir / image_path
    if relative_path.exists():
        return str(relative_path)
    
    # Try project root
    project_root = Path(__file__).parent
    project_path = project_root / image_path
    if project_path.exists():
        return str(project_path)
    
    # Try data/sample directory
    sample_path = project_root / "data" / "sample" / image_path
    if sample_path.exists():
        return str(sample_path)
    
    # Try just the filename in data/sample
    filename = Path(image_path).name
    sample_filename = project_root / "data" / "sample" / filename
    if sample_filename.exists():
        return str(sample_filename)
    
    # Try common ADK upload patterns
    adk_patterns = [
        f"/tmp/{image_path}",
        f"/var/tmp/{image_path}",
        f"/tmp/uploads/{image_path}",
        f"/var/tmp/uploads/{image_path}",
        f"{image_path}.pdf",
        f"{image_path}.png",
        f"{image_path}.jpg",
        f"{image_path}.jpeg",
    ]
    
    for pattern in adk_patterns:
        if os.path.exists(pattern):
            return pattern
    
    # Try to find recently uploaded files
    recent_files = find_recent_uploaded_files()
    if recent_files:
        # Look for files that match our expected name pattern
        target_name = Path(image_path).stem
        for rf in recent_files:
            rf_name = Path(rf).stem
            if target_name in rf_name or rf_name in target_name:
                return rf
        
        # If no exact match, use the most recent PDF or image file
        for rf in recent_files:
            if rf.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
                return rf
    
    # Last resort: use sample file if available
    sample_file = project_root / "data" / "sample" / "1800HistoryTestOCR.pdf"
    if sample_file.exists():
        return str(sample_file)
    
    return image_path

def analyze_document_with_multi_agent(image_path: str, question: str, question_types: str = ""):
    """
    Analyze a document using the multi-agent system with A2A communication.
    
    Args:
        image_path: Path to the document image (can be PDF, PNG, JPG, etc.)
        question: Question to ask about the document
        question_types: Comma-separated string of question types (e.g., "table/list,form")
    
    Returns:
        Dict containing the analysis result with A2A communication details
    """
    try:
        import os
        from pathlib import Path
        
        # Debug: Log the input parameters
        # Debug logging removed for production
        
        # Handle case where user mentions PDF in question but path doesn't exist
        if "pdf" in question.lower() and not os.path.exists(image_path):
            # First, try to find recently uploaded files
            recent_files = find_recent_uploaded_files()
            
            # Look for PDF files in recent uploads
            pdf_files = [f for f in recent_files if f.lower().endswith('.pdf')]
            
            # Also look for the exact filename in recent files
            target_filename = Path(image_path).name
            exact_matches = [f for f in recent_files if Path(f).name == target_filename]
            
            if exact_matches:
                resolved_path = exact_matches[0]  # Use the first match
            elif pdf_files:
                # Use the most recent PDF file
                most_recent_pdf = max(pdf_files, key=os.path.getmtime)
                resolved_path = most_recent_pdf
            else:
                # Fallback to sample file
                sample_file = Path(__file__).parent / "data" / "sample" / "1800HistoryTestOCR.pdf"
                if sample_file.exists():
                    resolved_path = str(sample_file)
                else:
                    resolved_path = image_path
        else:
            # Resolve the file path
            resolved_path = resolve_file_path(image_path, question)
        
        # Check for page ambiguity BEFORE any PDF conversion
        if resolved_path.lower().endswith('.pdf'):
            ambiguity_info = detect_page_ambiguity(question, resolved_path)
            if ambiguity_info.get("has_ambiguity"):
                mentioned_page = ambiguity_info["mentioned_page"]
                pdf_total = ambiguity_info["pdf_total_pages"]
                printed_pages = ambiguity_info["printed_pages_found"]
                
                clarification_message = f"""I detected a potential ambiguity with page {mentioned_page}. 

**Two possible interpretations:**
1. **PDF Page {mentioned_page}** (the {mentioned_page}{'st' if mentioned_page == 1 else 'nd' if mentioned_page == 2 else 'rd' if mentioned_page == 3 else 'th'} page of the PDF document)
2. **Printed Page {mentioned_page}** (a page that has "{mentioned_page}" printed on it)

**Found printed page numbers in the document:**
"""
                for pdf_page, printed_page in printed_pages[:5]:
                    clarification_message += f"- PDF page {pdf_page} contains printed page number {printed_page}\n"
                
                clarification_message += f"""
**Please clarify which you mean:**
- Say "PDF page {mentioned_page}" for the actual PDF page
- Say "printed page {mentioned_page}" for the page with "{mentioned_page}" printed on it
- Or rephrase your question to be more specific

**Example:** "Give me a summary of the paragraph on PDF page {mentioned_page}" or "What's on the page that has {mentioned_page} printed on it?"
"""
                
                return {
                    "status": "clarification_needed",
                    "answer": clarification_message,
                    "confidence": 1.0,
                    "ambiguity_info": ambiguity_info,
                    "agent_type": "page_ambiguity_detector"
                }
        
        # Ensure PDF is converted to image before processing
        if resolved_path.lower().endswith('.pdf'):
            page_number = extract_page_number_from_question(question) if question else 1
            resolved_path = convert_pdf_to_image(resolved_path, page_number)
        
        # Check if the file exists
        if not os.path.exists(resolved_path):
            # Try to suggest using the sample file
            sample_file = Path(__file__).parent / "data" / "sample" / "1800HistoryTestOCR.pdf"
            sample_suggestion = ""
            if sample_file.exists():
                sample_suggestion = f"\n\n💡 **Suggestion**: You can try using the sample file by asking: 'Analyze the sample document: 1800HistoryTestOCR.pdf'"
            
            return {
                "error": f"File not found: {image_path}",
                "status": "error",
                "answer": f"I cannot find the uploaded PDF file '{image_path}'. This is likely because:\n\n1. **ADK Upload Issue**: The file wasn't properly uploaded to the server\n2. **File Processing Delay**: ADK might still be processing the upload\n3. **Path Resolution**: The file path from ADK doesn't match our search locations\n\n**What I searched**:\n- Recent uploads in temp directories\n- ADK-specific upload locations\n- System temporary directories\n\n**Solutions**:\n- Try re-uploading the PDF file\n- Wait a moment and try again (file might still be processing)\n- Make sure the file is a valid PDF\n- Check that the file isn't corrupted\n\n{sample_suggestion}",
                "confidence": 0.0,
                "debug_info": {
                    "original_path": image_path,
                    "resolved_path": resolved_path,
                    "file_exists": False,
                    "current_directory": os.getcwd(),
                    "project_directory": str(Path(__file__).parent),
                    "available_files": os.listdir('.') if os.path.exists('.') else [],
                    "recent_files": find_recent_uploaded_files(),
                    "search_locations_checked": [
                        '.', '/tmp', '/var/tmp', '/tmp/uploads', '/var/tmp/uploads',
                        str(Path(__file__).parent), str(Path(__file__).parent / "uploads")
                    ]
                }
            }
        
        # Parse question_types string into list
        parsed_question_types = None
        if question_types and question_types.strip():
            parsed_question_types = [qtype.strip() for qtype in question_types.split(',')]
        
        result = multi_agent_wrapper.analyze_document_with_a2a(
            image_path=resolved_path,
            question=question,
            question_types=parsed_question_types,
            show_communication=True
        )
        return result
    except Exception as e:
        return {
            "error": str(e),
            "status": "error",
            "answer": f"An error occurred while processing your request: {str(e)}",
            "confidence": 0.0,
            "debug_info": {
                "exception_type": type(e).__name__,
                "exception_message": str(e)
            }
        }

def get_agent_info():
    """
    Get information about the multi-agent system capabilities.
    
    Returns:
        Dict containing agent information and capabilities
    """
    try:
        return multi_agent_wrapper.get_wrapper_info()
    except Exception as e:
        return {
            "error": str(e),
            "agents": [],
            "capabilities": []
        }

# Create the ADK agent
root_agent = Agent(
    name="multi_agent_document_analyzer",
    model=os.getenv('MODEL', 'gemini-2.5-flash'),
    description=(
        "A multi-agent AI system specialized in document analysis with intelligent routing. "
        "Uses Vision Specialist, OCR Specialist, Layout Specialist, and Answer Validator agents "
        "with A2A (Agent-to-Agent) communication for optimal results."
    ),
    instruction="""You are a Multi-Agent Document Analysis Expert with intelligent routing capabilities.

## Your Multi-Agent System:

**🎯 Orchestrator Agent**: Routes questions to the best specialist
**👁️ Vision Specialist**: Handles images, photos, figures, diagrams, handwritten text, Yes/No questions
**📊 OCR Specialist**: Handles tables, lists, forms, free text, structured data
**📐 Layout Specialist**: Handles layout analysis, abstract questions, document structure
**✅ Answer Validator**: Validates answers and provides confidence scoring

## How to Use:

### For Document Analysis:
1. **Upload your document** (PDF, PNG, JPG, receipt, invoice, form, etc.)
2. **Ask your question** - be specific about what you want to know
3. **The system will automatically route** to the best specialist agent
4. **Get validated results** with confidence scores and A2A communication details

**PDF Support**: The system automatically converts PDFs to images for analysis. Just mention "PDF" in your question and upload the PDF file.

### Question Types Supported:
- **Images/Photos**: "What do you see in this image?"
- **Tables/Lists**: "What are the line items in this table?"
- **Forms**: "What information is filled in this form?"
- **Layout**: "What is the structure of this document?"
- **Handwritten**: "What does this handwritten text say?"
- **Yes/No**: "Is there a signature on this document?"

### Question Types Parameter:
- Use comma-separated values: "table/list,form" or "Image/Photo" or "layout"
- Leave empty for auto-detection: ""

### Example Questions:
- "What is the total amount on this receipt?"
- "What are the line items in this invoice?"
- "What information is in this form?"
- "What does this diagram show?"
- "Is there a signature on this document?"
- "What is the caption of the first image on page 4?"

## Response Format:
The system will provide:
- **Answer**: The main response to your question
- **Confidence**: Confidence score (0.0 to 1.0)
- **Selected Agent**: Which specialist handled your question
- **A2A Communication**: Details of agent-to-agent communication
- **Processing Time**: How long the analysis took

## Tips:
- Be specific in your questions for better results
- Upload clear, high-quality images
- The system automatically chooses the best agent for your question type
- All agents work together with A2A communication for optimal results
- If you upload a PDF, specify the page number in your question (e.g., "page 4", "page four")
- **Page Number Clarification**: If there's ambiguity between PDF page numbers and printed page numbers, the system will ask for clarification

Ready to analyze your documents with enterprise-grade multi-agent intelligence! 🚀""",
    
    tools=[
        analyze_document_with_multi_agent,
        get_agent_info
    ]
)

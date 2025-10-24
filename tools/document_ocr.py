"""
Custom tools for document processing using Google Document AI
These will be used by the ADK agent
"""

from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions
import os
import json
import logging

logger = logging.getLogger(__name__)

# Global state for tool initialization
_ocr_processor = None

def initialize_ocr_processor():
    """Initialize Document AI processor (called once)"""
    global _ocr_processor
    if _ocr_processor is None:
        project_id = os.getenv('GCP_PROJECT_ID')
        location = os.getenv('GCP_LOCATION', 'us')
        processor_id = os.getenv('DOCUMENT_AI_PROCESSOR_ID')
        
        logger.info(f"Initializing Document AI processor in {location} (Project: {project_id})")
        
        opts = ClientOptions(
            api_endpoint=f"{location}-documentai.googleapis.com",
            client_cert_source=None
        )
        # Use REST transport instead of gRPC to avoid DNS issues
        try:
            client = documentai.DocumentProcessorServiceClient(
                client_options=opts,
                transport='rest'
            )
            logger.debug("Using REST transport for Document AI")
        except Exception as e:
            logger.debug(f"REST transport failed, falling back to gRPC: {e}")
            client = documentai.DocumentProcessorServiceClient(client_options=opts)
        
        processor_name = client.processor_path(project_id, location, processor_id)
        
        _ocr_processor = {
            'client': client,
            'processor_name': processor_name
        }
    
    return _ocr_processor


def process_document_with_ocr(document_path: str) -> str:
    """
    Process a document using Google Document AI OCR.
    
    Extracts text, tables, and entities from PDF or image files.
    Use this tool when the user asks to analyze a document, extract information
    from a PDF, or read a receipt/invoice/form.
    
    Args:
        document_path: Path to the document file (PDF or image). Can be relative (e.g., "data/sample/file.pdf") or absolute.
        
    Returns:
        JSON string containing extracted text, tables, and metadata
    """
    try:
        processor = initialize_ocr_processor()
        
        # Convert relative paths to absolute paths based on project root
        if not os.path.isabs(document_path):
            # Assume relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            document_path = os.path.join(project_root, document_path)
        
        # Check if file exists
        if not os.path.exists(document_path):
            error_result = {
                'status': 'error',
                'error_message': f"File not found at path: {document_path}. Please check the path and try again."
            }
            return json.dumps(error_result, indent=2)
        
        # Read document
        with open(document_path, "rb") as f:
            document_content = f.read()
        
        # Determine MIME type
        if document_path.endswith('.pdf'):
            mime_type = 'application/pdf'
        elif document_path.endswith(('.png', '.jpg', '.jpeg')):
            mime_type = f'image/{document_path.split(".")[-1]}'
        else:
            mime_type = 'application/pdf'
        
        # Process with Document AI
        raw_document = documentai.RawDocument(
            content=document_content,
            mime_type=mime_type
        )
        
        request = documentai.ProcessRequest(
            name=processor['processor_name'],
            raw_document=raw_document
        )
        
        logger.debug(f"Processing document: {len(document_content)} bytes ({mime_type})")
        
        result = processor['client'].process_document(request=request)
        document = result.document
        
        logger.info(f"Extracted {len(document.text)} characters from {len(document.pages)} page(s)")
        
        # Extract structured data
        extracted = {
            'status': 'success',
            'full_text': document.text,
            'num_pages': len(document.pages),
            'pages': [],
            'tables': [],
            'entities': []
        }
        
        # Extract page info
        for page_idx, page in enumerate(document.pages):
            page_info = {
                'page_number': page_idx + 1,
                'text_blocks': []
            }
            
            # Extract text blocks
            for block in page.blocks:
                if block.layout.text_anchor:
                    block_text = _extract_text_from_layout(block.layout, document.text)
                    if block_text.strip():
                        page_info['text_blocks'].append(block_text)
            
            extracted['pages'].append(page_info)
            
            # Extract tables
            for table in page.tables:
                table_data = _extract_table(table, document.text, page_idx + 1)
                extracted['tables'].append(table_data)
        
        # Extract entities (for specialized processors)
        for entity in document.entities:
            extracted['entities'].append({
                'type': entity.type_,
                'value': entity.mention_text,
                'confidence': entity.confidence
            })
        
        return json.dumps(extracted, indent=2)
        
    except Exception as e:
        logger.error(f"Document AI processing failed: {type(e).__name__}: {str(e)}", exc_info=True)
        error_result = {
            'status': 'error',
            'error_message': f"Failed to process document: {type(e).__name__}: {str(e)}"
        }
        return json.dumps(error_result, indent=2)


def search_document(document_path: str, query: str) -> str:
    """
    Search for relevant information in a document using hybrid retrieval.
    
    This tool first processes the document with OCR, then uses semantic search
    to find the most relevant sections for answering the query.
    
    Use this when the user asks a specific question about a document's content.
    
    Args:
        document_path: Path to the document file. Can be relative (e.g., "data/sample/file.pdf") or absolute.
        query: The question or search query
        
    Returns:
        JSON string with relevant text segments and their relevance scores
    """
    try:
        # Convert relative paths to absolute paths based on project root
        if not os.path.isabs(document_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            document_path = os.path.join(project_root, document_path)
        
        # First process with OCR (now with absolute path)
        ocr_result_json = process_document_with_ocr(document_path)
        ocr_result = json.loads(ocr_result_json)
        
        if ocr_result['status'] == 'error':
            return ocr_result_json
        
        # Simple keyword-based search (simplified for ADK demo)
        # In production, you'd use ColPali/ColBERT here
        query_lower = query.lower()
        relevant_segments = []
        
        for page in ocr_result['pages']:
            for block in page['text_blocks']:
                # Simple relevance scoring based on keyword overlap
                if any(word in block.lower() for word in query_lower.split()):
                    relevant_segments.append({
                        'text': block,
                        'page': page['page_number'],
                        'relevance': sum(word in block.lower() for word in query_lower.split())
                    })
        
        # Sort by relevance
        relevant_segments.sort(key=lambda x: x['relevance'], reverse=True)
        
        result = {
            'status': 'success',
            'query': query,
            'num_results': len(relevant_segments[:4]),
            'results': relevant_segments[:4],  # Top 4 results
            'full_document_text': ocr_result['full_text'][:2000]  # Preview
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            'status': 'error',
            'error_message': f"Search failed: {str(e)}"
        }
        return json.dumps(error_result, indent=2)


def _extract_text_from_layout(layout, full_text: str) -> str:
    """Helper to extract text from layout object"""
    if not layout.text_anchor or not layout.text_anchor.text_segments:
        return ""
    
    text_segments = []
    for segment in layout.text_anchor.text_segments:
        start_idx = int(segment.start_index) if segment.start_index else 0
        end_idx = int(segment.end_index) if segment.end_index else len(full_text)
        text_segments.append(full_text[start_idx:end_idx])
    
    return "".join(text_segments)


def _extract_table(table, full_text: str, page_number: int) -> dict:
    """Helper to extract table structure"""
    table_data = {
        'page': page_number,
        'headers': [],
        'rows': []
    }
    
    # Extract headers
    if table.header_rows:
        for cell in table.header_rows[0].cells:
            cell_text = _extract_text_from_layout(cell.layout, full_text)
            table_data['headers'].append(cell_text.strip())
    
    # Extract rows
    for row in table.body_rows:
        row_data = []
        for cell in row.cells:
            cell_text = _extract_text_from_layout(cell.layout, full_text)
            row_data.append(cell_text.strip())
        table_data['rows'].append(row_data)
    
    return table_data

from urllib.parse import urlparse
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask import Flask, redirect, render_template, request, jsonify, session, send_file, flash, url_for, Response
from azure.storage.blob import generate_blob_sas, BlobSasPermissions, BlobServiceClient
from datetime import datetime, timedelta
from github_doc_generator import GitHubDocGenerator
from github_doc_generator import GitHubDocGenerator, DocumentationConfig
import pyodbc
from werkzeug.utils import secure_filename
import json
import os
from urllib.parse import urlparse, unquote
import markdown
import re
import requests
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from PIL import Image
# Add to your imports
from PIL import Image, ImageFilter  # Add ImageFilter here
from io import BytesIO
import base64
import tempfile
import zipfile
import shutil
import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    SearchableField
)
 
 
app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)  # More secure secret key generation
 
# Define the default configuration at the module level
DEFAULT_CONFIG = {
    'include_overview': True,
    'include_executive_summary': True,
    'include_tech_stack': True,
    'include_code_structure': True,
    'include_loc_analysis': True,
    'include_loc_chart': True,
    'include_complexity_analysis': True,
    'include_complexity_charts': True,
    'include_features': True,
    'include_dependencies': True,
    'include_issues': True,
    'include_sql_objects': True,
    'include_class_diagram': True,
    'include_flow_diagram': True,
    'include_er_diagram': True,
    'include_reference_architecture': True,
    'max_files_to_analyze': 5
}
 
# Configuration
SAVES_DIR = 'saved_diagrams'
if not os.path.exists(SAVES_DIR):
    os.makedirs(SAVES_DIR)
 
# Configure output directory for generated docs
OUTPUT_DIR = Path('generated_docs')
OUTPUT_DIR.mkdir(exist_ok=True)
 
# Load configuration from environment variables
app.config['GITHUB_TOKEN'] = "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"
DEPLOYMENT_NAME = "docwriter-gpt-4o"
API_KEY = "DmeYPgsWhvBuyttBZMQZsUM7GAW0QAn6pWfxYara20dwZqxH0ylrJQQJ99BHACHYHv6XJ3w3AAAAACOG5bWx"
API_BASE = "https://docwriter.cognitiveservices.azure.com/"
 
 
# Azure Blob Storage Configuration
app.config['AZURE_STORAGE_CONNECTION_STRING'] = "DefaultEndpointsProtocol=https;AccountName=docwritertool;AccountKey=ckTIdmfnZoRz7w9Xmwp6pxztPFSL/mH9VAVGA8QyuXwtuutxBOSLT1hOQqNEQWUJ1sjU3eRJDa70+AStBp9y6Q==;EndpointSuffix=core.windows.net"
app.config['AZURE_STORAGE_CONTAINER'] = "docwriter-container"
app.config['AZURE_STORAGE_ACCOUNT'] = "docwritertool"
app.config['AZURE_STORAGE_KEY'] = "ckTIdmfnZoRz7w9Xmwp6pxztPFSL/mH9VAVGA8QyuXwtuutxBOSLT1hOQqNEQWUJ1sjU3eRJDa70+AStBp9y6Q=="
 
# Azure SQL Configuration
app.config['SQL_SERVER'] = 'ycatserver.database.windows.net'
app.config['SQL_DATABASE'] = 'ycat'
app.config['SQL_USERNAME'] = 'ycattool_admin'
app.config['SQL_PASSWORD'] = 'Cat-db@123'  
app.config['SQL_DRIVER'] = 'ODBC Driver 17 for SQL Server'

# Azure Cognitive Search Configuration
app.config['AZURE_SEARCH_SERVICE_NAME'] = "docwritersearchai"
app.config['AZURE_SEARCH_ADMIN_KEY'] = "P00ONTx9LLtiuQSGylkCUfLPwGqGnRoSXP5CGnaPZhAzSeB631Qq"
app.config['AZURE_SEARCH_INDEX_NAME'] = "docs-index"

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
 
# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, email, name):
        self.id = id
        self.email = email
        self.name = name
        self._documents = None
 
    @property
    def documents(self):
        if self._documents is None:
            conn = None
            cursor = None
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
            
                # SQL Server compatible query using TOP instead of LIMIT
                query = """
                    SELECT
                        id,
                        document_name AS name,
                        document_type AS type,
                        blob_url,
                        created_at,
                        repo_name
                    FROM documents
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                """
            
                cursor.execute(query, (self.id,))
            
                # Get column names from cursor description
                columns = [column[0] for column in cursor.description]
            
                # Convert rows to dictionaries
                self._documents = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            except Exception as e:
                # Log the error or handle it appropriately
                print(f"Error fetching documents: {str(e)}")
                self._documents = []  # Return empty list on error
            finally:
                # Ensure resources are properly closed
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()
                
        return self._documents or []

    @staticmethod
    def init_search_index():
        service_name = app.config['AZURE_SEARCH_SERVICE_NAME']
        admin_key = app.config['AZURE_SEARCH_ADMIN_KEY']
        index_name = app.config['AZURE_SEARCH_INDEX_NAME']
        endpoint = f"https://{service_name}.search.windows.net"
        credential = AzureKeyCredential(admin_key)
        
        index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
        
        fields = [
            SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="repo_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="document_name", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="document_type", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
            SimpleField(name="blob_url", type=SearchFieldDataType.String),
            SimpleField(name="user_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="page_number", type=SearchFieldDataType.Int32),
            SimpleField(name="content_type", type=SearchFieldDataType.String),
            SimpleField(name="image_base64", type=SearchFieldDataType.String),
            SimpleField(name="is_graph", type=SearchFieldDataType.Boolean)
        ]
        
        index = SearchIndex(name=index_name, fields=fields)
        try:
            index_client.create_or_update_index(index)
        except Exception as e:
            print(f"Index error: {e}")

    # Initialize index when app starts
    init_search_index()

def delete_repo_from_index(repo_name):
    service_name = app.config['AZURE_SEARCH_SERVICE_NAME']
    admin_key = app.config['AZURE_SEARCH_ADMIN_KEY']
    index_name = app.config['AZURE_SEARCH_INDEX_NAME']
    endpoint = f"https://{service_name}.search.windows.net"
    credential = AzureKeyCredential(admin_key)
    
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
    
    try:
        results = search_client.search(search_text="*", filter=f"repo_name eq '{repo_name}'")
        documents_to_delete = [{"chunk_id": doc["chunk_id"]} for doc in results]
        
        if documents_to_delete:
            search_client.delete_documents(documents=documents_to_delete)
    except Exception as e:
        app.logger.error(f"Error deleting from index: {str(e)}")

# Add these functions BEFORE the route that uses them
def chunk_text(text, chunk_size=1000, overlap=200):
    """Context-aware text chunking with sentence boundaries"""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sent_len = len(sentence)
        if current_length + sent_len > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[-int(len(current_chunk) * overlap / chunk_size):]
            current_length = sum(len(s) for s in current_chunk)
        current_chunk.append(sentence)
        current_length += sent_len
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def extract_text_from_pdf(pdf_path):
    """Extract text, images, tables and metadata from PDF with PyMuPDF and pdfplumber"""
    content = {
        'text': [],
        'images': [],
        'tables': [],
        'graphs': [],
        'metadata': {}
    }

    try:
        # Extract text and images with PyMuPDF
        with fitz.open(pdf_path) as doc:
            content['metadata'] = {
                'author': doc.metadata.get('author', ''),
                'title': doc.metadata.get('title', os.path.basename(pdf_path)),
                'pages': len(doc)
            }

            for page_num, page in enumerate(doc, 1):
                # Extract text
                text = page.get_text()
                content['text'].append({
                    'page_number': page_num,
                    'text': text,
                    'blocks': [b[4] for b in page.get_text("blocks")]
                })

                # Extract images
                img_list = page.get_images(full=True)
                for img_index, img in enumerate(img_list):
                    base_image = doc.extract_image(img[0])
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    pil_image = Image.open(BytesIO(image_bytes))
                    is_graph = analyze_image_for_graph(pil_image)
                    
                    # Convert to base64
                    buffered = BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    content['images'].append({
                        'page_number': page_num,
                        'image_index': img_index,
                        'base64_image': img_base64,
                        'is_graph': is_graph
                    })

        # Extract tables with pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                for table_num, table in enumerate(tables, 1):
                    if table:
                        content['tables'].append({
                            'page_number': page_num,
                            'table_num': table_num,
                            'data': [[cell.strip() if cell else "" for cell in row] for row in table]
                        })

        return content
    except Exception as e:
        app.logger.error(f"PDF Extraction Error: {str(e)}")
        raise


def analyze_image_for_graph(pil_image):
    """Analyze image to detect graphs/charts using edge detection"""
    try:
        # Convert to grayscale
        grayscale = pil_image.convert('L')
        
        # Edge detection with proper import
        edges = grayscale.filter(ImageFilter.FIND_EDGES)
        
        # Rest of the function remains the same...
        edge_pixels = sum(1 for p in edges.getdata() if p > 200)
        total_pixels = edges.width * edges.height
        edge_percent = (edge_pixels / total_pixels) * 100
        
        return 5 < edge_percent < 30
    except Exception as e:
        app.logger.error(f"Graph detection error: {e}")
        return False


# Updated index_document_chunks function (standalone version)
def index_document_chunks(repo_name, document_name, document_type, content, blob_url, user_id):
    """Index document content with page-wise chunks and visual elements"""
    service_name = app.config['AZURE_SEARCH_SERVICE_NAME']
    admin_key = app.config['AZURE_SEARCH_ADMIN_KEY']
    index_name = app.config['AZURE_SEARCH_INDEX_NAME']
    
    search_client = SearchClient(
        endpoint=f"https://{service_name}.search.windows.net",
        index_name=index_name,
        credential=AzureKeyCredential(admin_key)
    )

    # First, delete any existing chunks for this repo/document combination
    try:
        # Search for existing chunks for this repo/document
        results = search_client.search(
            search_text="*",
            filter=f"repo_name eq '{repo_name}' and document_name eq '{document_name}' and user_id eq '{user_id}'"
        )
        
        # Collect all existing chunk IDs to delete
        documents_to_delete = [{"chunk_id": doc["chunk_id"]} for doc in results]
        
        if documents_to_delete:
            search_client.delete_documents(documents=documents_to_delete)
            app.logger.info(f"Deleted {len(documents_to_delete)} existing chunks for {document_name}")
    except Exception as e:
        app.logger.error(f"Error deleting existing chunks: {str(e)}")
        # Continue with indexing even if deletion fails

    documents = []
    
    # Index text chunks (updated to include repo_name in chunk_id)
    for page in content['text']:
        page_chunks = chunk_text(page['text'])
        for chunk_num, chunk in enumerate(page_chunks):
            doc_id = f"{user_id}-{repo_name}-{page['page_number']}-{chunk_num}"
            documents.append({
                "chunk_id": doc_id,
                "repo_name": repo_name,
                "document_name": document_name,
                "document_type": document_type,
                "content": chunk,
                "blob_url": blob_url,
                "user_id": str(user_id),
                "page_number": page['page_number'],
                "content_type": "text"
            })

    # Index images (updated to include repo_name in chunk_id)
    for img in content['images']:
        doc_id = f"{user_id}-{repo_name}-{img['page_number']}-img-{img['image_index']}"
        documents.append({
            "chunk_id": doc_id,
            "repo_name": repo_name,
            "document_name": document_name,
            "document_type": document_type,
            "content": "Image content",
            "blob_url": blob_url,
            "user_id": str(user_id),
            "page_number": img['page_number'],
            "content_type": "image",
            "image_base64": img['base64_image'],
            "is_graph": img['is_graph']
        })

    # Index tables
    for table in content['tables']:
        doc_id = f"{user_id}-{repo_name}-{table['page_number']}-tbl-{table['table_num']}"
        table_content = "\n".join(["|".join(row) for row in table['data']])
        documents.append({
            "chunk_id": doc_id,
            "repo_name": repo_name,
            "document_name": document_name,
            "document_type": document_type,
            "content": f"Table data:\n{table_content}",
            "blob_url": blob_url,
            "user_id": str(user_id),
            "page_number": table['page_number'],
            "content_type": "table"
        })

    if documents:
        try:
            result = search_client.upload_documents(documents=documents)
            app.logger.info(f"Indexed {len(documents)} chunks for {document_name} in repo {repo_name}")
        except Exception as e:
            app.logger.error(f"Indexing error: {str(e)}")
            raise

# Updated chat handler route
@app.route('/chat', methods=['POST'])
@login_required
def chat_handler():
    try:
        data = request.json
        question = data.get('question', '').strip().lower()
        conversation_history = data.get('history', [])  # Get conversation history from frontend
        repo_name = data.get('repo_name', '') 

        if not question:
            return jsonify({'error': 'Please ask a question'}), 400
        
    
        # Add the new user message to the conversation history
        conversation_history.append({"role": "user", "content": question})

        search_client = SearchClient(
            endpoint=f"https://{app.config['AZURE_SEARCH_SERVICE_NAME']}.search.windows.net",
            index_name=app.config['AZURE_SEARCH_INDEX_NAME'],
            credential=AzureKeyCredential(app.config['AZURE_SEARCH_ADMIN_KEY'])
        )

        # Add repo filter if a repo is selected
        filter_query = f"user_id eq '{current_user.id}'"
        if repo_name:
            filter_query += f" and repo_name eq '{repo_name}'"

        results = search_client.search(
            search_text=question,
            filter=filter_query,
            top=10,
            include_total_count=True,
            highlight_pre_tag="<mark>",
            highlight_post_tag="</mark>",
            select=["content", "document_name", "blob_url", "repo_name", "page_number", "content_type", "image_base64"]
        )

        context = []
        seen = set()
        
        for result in results:
            try:
                doc_key = f"{result['document_name']}-{result['page_number']}"
                if doc_key in seen:
                    continue
                seen.add(doc_key)

                content = result.get('content', '')
                highlighted = result.get('@search.highlights')
                if highlighted and 'content' in highlighted:
                    content = highlighted['content'][0]

                # Generate SAS token for secure access
                parsed_url = urlparse(result['blob_url'])
                blob_path = unquote(parsed_url.path).lstrip('/')
                sas_token = generate_blob_sas(
                    account_name=app.config['AZURE_STORAGE_ACCOUNT'],
                    container_name=app.config['AZURE_STORAGE_CONTAINER'],
                    blob_name=blob_path,
                    account_key=app.config['AZURE_STORAGE_KEY'],
                    permission=BlobSasPermissions(read=True),
                    expiry=datetime.utcnow() + timedelta(minutes=15)
                )

                context.append({
                    'content': content,
                    'source': result['blob_url'],
                    'repo': result['repo_name'],
                    'page': result['page_number'],
                    'type': result.get('content_type', 'text'),
                    'image': result.get('image_base64'),
                    'url': f"{result['blob_url']}"
                })

            except Exception as e:
                app.logger.error(f"Result processing error: {str(e)}")
                continue

        if not context:
            return jsonify({
                'answer': "No relevant information found in documents.",
                'history': conversation_history  # Return updated history
            })

        # Generate enhanced response with OpenAI using conversation history
        openai.api_type = "azure"
        openai.api_base = "https://docwriter.cognitiveservices.azure.com/"
        openai.api_version = "2024-05-01-preview"
        openai.api_key = "DmeYPgsWhvBuyttBZMQZsUM7GAW0QAn6pWfxYara20dwZqxH0ylrJQQJ99BHACHYHv6XJ3w3AAAAACOG5bWx"

        # Prepare messages with system prompt and conversation history
        messages = [{
            "role": "system",
            "content": """You are DocWriter AI, an expert documentation assistant specialized in analyzing and explaining technical documentation. 
            Your responses should be:

            1. **Context-Aware**:
            - Maintain context from previous messages in the conversation
            - Reference earlier discussions when relevant
            - Ask clarifying questions if the request is ambiguous

            2. **Technical Accuracy**:
            - Only provide information that can be verified in the provided context
            - If unsure, say "I couldn't find definitive information about this in your documents"
            - For technical terms, provide clear explanations

            3. **Structured Responses**:
            - Use markdown formatting for clear presentation
            - Break complex explanations into bullet points
            - Use tables for comparative information
            - Include code blocks with syntax highlighting when appropriate

            4. **Citation**:
            - Always cite your sources from the documents when possible
            - Include page numbers or document names
            - For visual elements, describe key insights from charts/diagrams

            5. **User Focus**:
            - Adapt explanations to the user's apparent technical level
            - Provide both high-level overviews and detailed technical details when requested
            - Offer multiple perspectives when appropriate

            6. **Safety**:
            - Never hallucinate information not present in the documents
            - Flag potential inconsistencies in the source material
            - Disclose confidence level when appropriate
"""
        }]

        # Add conversation history
        messages.extend(conversation_history[-6:])  # Keep last 6 messages for context

        # Add current context
        messages.append({
            "role": "user",
            "content": f"Context:\n{json.dumps(context, indent=2)}\n\nQuestion: {question}"
        })

        response = openai.ChatCompletion.create(
            engine="docwriter-gpt-4o",
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )

        answer = response.choices[0].message.content
        
        # Add assistant response to conversation history
        conversation_history.append({"role": "assistant", "content": answer})
        
        return jsonify({
            'answer': answer,
            'context': context,
            'history': conversation_history  # Return updated history
        })

    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': 'Error processing request'}), 500


# Add this test route to check index contents
@app.route('/debug-search', methods=['GET'])
@login_required
def debug_search():
    service_name = app.config['AZURE_SEARCH_SERVICE_NAME']
    admin_key = app.config['AZURE_SEARCH_ADMIN_KEY']
    index_name = app.config['AZURE_SEARCH_INDEX_NAME']
    
    search_client = SearchClient(
        endpoint=f"https://{service_name}.search.windows.net",
        index_name=index_name,
        credential=AzureKeyCredential(admin_key)
    )
    
    try:
        results = search_client.search(search_text="*", filter=f"user_id eq '{current_user.id}'")
        return jsonify([dict(r) for r in results])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Initialize session configurations
@app.before_request
def before_request():
    # Initialize configuration for new sessions
    if 'user_config' not in session and request.endpoint != 'static':
        session['user_config'] = DEFAULT_CONFIG.copy()
 
# Then add the user_loader
@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()  # Use your existing connection function
    cursor = conn.cursor()
   
    try:
        cursor.execute("SELECT id, email, name FROM users WHERE id = ?", user_id)
        user_data = cursor.fetchone()
       
        if user_data:
            return User(id=user_data[0], email=user_data[1], name=user_data[2])
        return None
    finally:
        conn.close()
 
# Database connection function
def get_db_connection():
    connection_string = (
        f"DRIVER={{{app.config['SQL_DRIVER']}}};"
        f"SERVER={app.config['SQL_SERVER']};"
        f"DATABASE={app.config['SQL_DATABASE']};"
        f"UID={app.config['SQL_USERNAME']};"
        f"PWD={app.config['SQL_PASSWORD']};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )
    try:
        return pyodbc.connect(connection_string)
    except pyodbc.Error as e:
        app.logger.error(f"Database connection failed: {str(e)}")
        raise

# Azure Blob Storage Helper Functions
def upload_to_blob_storage(file_path, user_id, document_type, repo_name=None):
    blob_service_client = BlobServiceClient.from_connection_string(
        app.config['AZURE_STORAGE_CONNECTION_STRING']
    )
    container_client = blob_service_client.get_container_client(
        app.config['AZURE_STORAGE_CONTAINER']
    )
   
    if repo_name:
        blob_name = f"{user_id}/{repo_name}/{os.path.basename(file_path)}"
    else:
        blob_name = f"{user_id}/{os.path.basename(file_path)}"
   
    with open(file_path, "rb") as data:
        container_client.upload_blob(name=blob_name, data=data, overwrite=True)
   
    return f"https://{blob_service_client.account_name}.blob.core.windows.net/{app.config['AZURE_STORAGE_CONTAINER']}/{blob_name}"
# Initialize Blob Service Client
blob_service_client = BlobServiceClient.from_connection_string(app.config['AZURE_STORAGE_CONNECTION_STRING'])
 
def generate_sas_url(blob_name):
    # blob_name now could be either "user_id/repo_name/type/filename" or "user_id/type/filename"
    sas_token = generate_blob_sas(
        account_name=app.config['AZURE_STORAGE_ACCOUNT'],
        container_name=app.config['AZURE_STORAGE_CONTAINER'],
        blob_name=blob_name,
        account_key=app.config['AZURE_STORAGE_KEY'],
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(minutes=5)
    )
    return f"https://{app.config['AZURE_STORAGE_ACCOUNT']}.blob.core.windows.net/{app.config['AZURE_STORAGE_CONTAINER']}/{blob_name}?{sas_token}"
 
def generate_sas_token(blob_name):
    sas_token = generate_blob_sas(
        account_name=app.config['AZURE_STORAGE_ACCOUNT'],
        account_key=app.config['AZURE_STORAGE_KEY'],
        container_name=app.config['AZURE_STORAGE_CONTAINER'],
        blob_name=blob_name,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(minutes=5)
    )
    return sas_token
 
 
@app.route('/clear-docs', methods=['POST'])
@login_required
def clear_docs():
    # Remove documentation from session when explicitly requested
    if 'documentation' in session:
        del session['documentation']
    return jsonify({'status': 'success'})
 
def ensure_instance_folder():
    """Ensure the instance folder exists"""
    instance_path = Path(app.instance_path)
    instance_path.mkdir(parents=True, exist_ok=True)
    return instance_path
 
# Initialize paths
instance_path = ensure_instance_folder()
app.config['CONFIG_PATH'] = str(instance_path / "config.json")
app.config['CONFIG_PATH'] = os.path.join(app.instance_path, 'config.json')
 
class DiagramValidator:
    @staticmethod
    def validate_syntax(code):
        if not code.strip().startswith('classDiagram'):
            return False, "Diagram must start with 'classDiagram'"
       
        if code.count('{') != code.count('}'):
            return False, "Unbalanced braces in diagram"
       
        class_pattern = r'class\s+[A-Za-z_][A-Za-z0-9_]*\s*{'
        if not re.search(class_pattern, code):
            return False, "No valid class definitions found"
       
         # Allow empty classes
        class_pattern = r'class\s+[A-Za-z_][A-Za-z0-9_]*\s*(\{[^}]*\})?'
        if not re.search(class_pattern, code):
            return False, "No valid class definitions found"
       
        return True, "Diagram syntax is valid"
 
def is_valid_github_url(url):
    """Validate GitHub URL format"""
    # More flexible pattern that accepts various GitHub URL formats
    pattern = r'^https?:\/\/(?:www\.)?github\.com\/[\w-]+\/[\w.-]+\/?(?:\/?|(?:\.git))$'
    return bool(re.match(pattern, url))
 
@app.route('/view_document/<int:doc_id>')
@login_required
def view_document(doc_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, document_name, blob_url, document_type, repo_name
        FROM documents
        WHERE id = ? AND user_id = ?
    """, (doc_id, current_user.id))
   
    doc_data = cursor.fetchone()
    conn.close()
   
    if not doc_data:
        flash("Document not found", "danger")
        return redirect(url_for('user_documents'))
 
    try:
        doc_id, doc_name, blob_url, doc_type, repo_name = doc_data
       
        # Parse the blob URL to get the path components
        from urllib.parse import urlparse
        parsed_url = urlparse(blob_url)
        path = parsed_url.path.lstrip('/')
        parts = path.split('/')
        container_name = parts[0]
        blob_path = '/'.join(parts[1:])
       
        # Generate new SAS token for the blob
        sas_token = generate_sas_token(blob_path)
        secure_url = f"{blob_url.split('?')[0]}?{sas_token}"
       
        # Handle different document types
        if doc_type.lower() in ['markdown', 'txt','class_diagram','flow_diagram','reference_architecture','er_diagram']:
            response = requests.get(secure_url)
            if response.status_code == 200:
                if doc_type.lower() == 'markdown':
                    content = markdown.markdown(response.text)
                    return render_template('view_markdown.html',
                                         content=content,
                                         doc_name=doc_name)
                else:
                    return render_template('view_text.html',
                                         content=response.text,
                                         doc_name=doc_name)
       
        flash("This document type can't be viewed directly. Please download it.", "info")
        return redirect(url_for('download_document', doc_id=doc_id))
           
    except Exception as e:
        app.logger.error(f"View error: {e}")
        flash(f"Error displaying document: {str(e)}", "danger")
        return redirect(url_for('user_documents'))
 
 
@app.route('/repository/<repo_name>')
@login_required
def repo_documents(repo_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Verify the repository belongs to the current user
    cursor.execute("""
        SELECT COUNT(*) 
        FROM documents 
        WHERE user_id = ? AND repo_name = ?
    """, (current_user.id, repo_name))
    
    if cursor.fetchone()[0] == 0:
        conn.close()
        flash("Repository not found", "danger")
        return redirect(url_for('user_documents'))
    
    # Get all documents for this repository and user
    cursor.execute("""
        SELECT id, document_name, blob_url, document_type 
        FROM documents 
        WHERE user_id = ? AND repo_name = ?
        ORDER BY document_name
    """, (current_user.id, repo_name))
    
    documents = cursor.fetchall()
    conn.close()
    
    # Initialize paths and IDs as None
    markdown_path = None
    pdf_path = None
    class_diagram_path = None
    flow_diagram_path = None
    reference_architecture_path = None
    markdown_id = None
    pdf_id = None
    flow_diagram_id = None
    class_diagram_id = None
    reference_architecture_id = None
    
    # Process documents to set specific paths
    for doc in documents:
        doc_id, doc_name, blob_url, doc_type = doc
        if doc_type.lower() == 'markdown':
            markdown_path = blob_url
            markdown_id = doc_id
        elif doc_type.lower() == 'pdf':
            pdf_path = blob_url
            pdf_id = doc_id
        elif doc_type.lower() == 'class_diagram':
            class_diagram_path = blob_url
            class_diagram_id = doc_id
        elif doc_type.lower() == 'flow_diagram':
            flow_diagram_path = blob_url
            flow_diagram_id = doc_id
        elif doc_type.lower() == 'reference_architecture':
            reference_architecture_path = blob_url
            reference_architecture_id = doc_id
    
    # Get the current configuration
    config = session.get('user_config', DEFAULT_CONFIG.copy())
    
    return render_template('blobresult.html', 
                         repo_name=repo_name,
                         markdown_path=markdown_path,
                         pdf_path=pdf_path,
                         class_diagram_path=class_diagram_path,
                         flow_diagram_path=flow_diagram_path,
                         reference_architecture_path=reference_architecture_path,
                         markdown_id=markdown_id,
                         pdf_id=pdf_id,
                         class_diagram_id=class_diagram_id,
                         flow_diagram_id=flow_diagram_id,
                         reference_architecture_id=reference_architecture_id)



@app.route('/download_document/<int:doc_id>')
@login_required
def download_document(doc_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT document_name, blob_url, document_type
        FROM documents
        WHERE id = ? AND user_id = ?
    """, (doc_id, current_user.id))
   
    doc_data = cursor.fetchone()
    conn.close()
   
    if not doc_data:
        flash("Document not found", "danger")
        return redirect(url_for('user_documents'))
 
    doc_name, blob_url, doc_type = doc_data
   
    try:
        # Generate a SAS token for secure download
        from urllib.parse import urlparse
        parsed_url = urlparse(blob_url)
        path = parsed_url.path.lstrip('/')
        parts = path.split('/')
        container_name = parts[0]
        blob_name = '/'.join(parts[1:])
       
        # Get the blob client
        blob_client = BlobServiceClient.from_connection_string(
            app.config['AZURE_STORAGE_CONNECTION_STRING']
        ).get_blob_client(container=container_name, blob=blob_name)
       
        # Stream the blob content
        stream = blob_client.download_blob()
       
        # Use the original filename without appending type
        filename = secure_filename(doc_name)
           
        mime_types = {
            'pdf': 'application/pdf',
            'md': 'text/markdown',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'txt': 'text/plain'
        }
        mime_type = mime_types.get(doc_type.lower(), 'application/octet-stream')
       
        return Response(
            stream.readall(),
            mimetype=mime_type,
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"'
            }
        )
           
    except Exception as e:
        app.logger.error(f"Download error: {e}")
        flash(f"Unable to download document: {str(e)}", "danger")
        return redirect(url_for('user_documents'))
   
   
# Routes from the first application
@app.route('/diagram-editor')
def diagram_editor():
    return render_template('liveditor.html')
 
 
 
# Login routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
   
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
       
        if not email or not password:
            flash('Please enter both email and password', 'error')
            return render_template('login.html')
       
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, password, name FROM users WHERE email = ? AND password = ?", email,password)
        user_data = cursor.fetchone()
        conn.close()
       
        if user_data:
            user = User(id=user_data[0], email=user_data[1], name=user_data[3])
            login_user(user)
           
            # Update last login time
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET last_login = GETDATE() WHERE id = ?", user.id)
            conn.commit()
            conn.close()
           
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid email or password', 'error')
   
    return render_template('login.html')
 
@app.route('/logout')
@login_required
def logout():
    # Clear all session data including configurations
    session.clear()
    logout_user()
    return redirect(url_for('login'))
 
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
   
    if request.method == 'POST':
        email = request.form.get('email').strip()
        name = request.form.get('name').strip()
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
       
        # Validate all fields are filled
        if not all([email, name, password, confirm_password]):
            flash('Please fill all fields', 'error')
            return render_template('register.html')
       
        # Validate email format
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash('Please enter a valid email address', 'error')
            return render_template('register.html')
       
        # Validate name length
        if len(name) < 2:
            flash('Name must be at least 2 characters', 'error')
            return render_template('register.html')
       
        # Validate password match
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
       
        # Validate password complexity
        password_regex = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
        if not re.match(password_regex, password):
            flash('Password must be at least 8 characters with one uppercase, one lowercase, one number, and one special character.', 'error')
            return render_template('register.html')
       
        conn = get_db_connection()
        cursor = conn.cursor()
       
        try:
            # Check if email already exists
            cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
            if cursor.fetchone():
                flash('Email already registered', 'error')
                return render_template('register.html')
           
            # Create new user (STORING PLAIN TEXT PASSWORD - INSECURE)
            cursor.execute(
                "INSERT INTO users (email, name, password) VALUES (?, ?, ?)",
                (email, name, password)  # Storing plain text password
            )
            conn.commit()
           
            flash('Registration successful! Please login', 'success')
            return redirect(url_for('login'))
           
        except Exception as e:
            conn.rollback()
            flash('An error occurred during registration. Please try again.', 'error')
            return render_template('register.html')
           
        finally:
            conn.close()
   
    return render_template('register.html')
 
@app.route('/generate-diagram', methods=['POST'])
def generate_diagram():
    try:
        data = request.json
        diagram_code = data.get('code', '')
        theme = data.get('theme', 'default')
       
        # Validate diagram syntax
        is_valid, message = DiagramValidator.validate_syntax(diagram_code)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': message
            }), 400
       
        return jsonify({
            'status': 'success',
            'code': diagram_code,
            'theme': theme
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
 
 
@app.route('/save-diagram', methods=['POST'])
def save_diagram():
    try:
        data = request.json
        name = data.get('name', '').strip()
        if not name:
            name = f"diagram_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
       
        # Sanitize filename
        safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        filename = f"{safe_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(SAVES_DIR, filename)
       
        # Add metadata to saved diagram
        save_data = {
            'name': name,
            'code': data.get('code', ''),
            'theme': data.get('theme', 'default'),
            'created_at': datetime.now().isoformat(),
            'version': '2.0'  # For future compatibility
        }
       
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
           
        return jsonify({
            'status': 'success',
            'message': 'Diagram saved successfully',
            'filename': filename
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
 
@app.route('/load-diagrams', methods=['GET'])
def load_diagrams():
    try:
        diagrams = []
        for filename in os.listdir(SAVES_DIR):
            if filename.endswith('.json'):
                with open(os.path.join(SAVES_DIR, filename), 'r') as f:
                    diagram = json.load(f)
                    diagrams.append({
                        'filename': filename,
                        'name': diagram.get('name', filename),
                        'created_at': diagram.get('created_at'),
                        'theme': diagram.get('theme', 'default'),
                        'version': diagram.get('version', '1.0')
                    })
       
        # Sort diagrams by creation date, newest first
        diagrams.sort(key=lambda x: x.get('created_at', ''), reverse=True)
       
        return jsonify({
            'status': 'success',
            'diagrams': diagrams
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
 
@app.route('/load-diagram/<filename>', methods=['GET'])
def load_diagram(filename):
    try:
        filepath = os.path.join(SAVES_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({
                'status': 'error',
                'message': 'Diagram not found'
            }), 404
           
        with open(filepath, 'r') as f:
            diagram = json.load(f)
           
        return jsonify({
            'status': 'success',
            'diagram': diagram
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
 
@app.route('/delete-diagram/<filename>', methods=['DELETE'])
def delete_diagram(filename):
    try:
        filepath = os.path.join(SAVES_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({
                'status': 'error',
                'message': 'Diagram not found'
            }), 404
           
        os.remove(filepath)
        return jsonify({
            'status': 'success',
            'message': 'Diagram deleted successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
 
# Routes from the second application
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        github_link = request.form.get('github_link', '').strip()
        zip_file = request.files.get('zip_upload')
        temp_dir = None  # Initialize temp_dir for cleanup

        try:
            # Validate input
            if not github_link and not zip_file:
                flash("Please provide either a GitHub URL or upload a ZIP file", "error")
                return render_template('index.html')
            
            if github_link and zip_file:
                flash("Please provide only one input method (GitHub URL or ZIP file)", "error")
                return render_template('index.html')

            # Initialize variables
            repo_name = None
            local_repo_path = None

            if github_link:
                # Process GitHub URL
                if not is_valid_github_url(github_link):
                    flash("Invalid GitHub URL format", "error")
                    return render_template('index.html')
                
                # Extract repository name from GitHub URL
                repo_name = github_link.split('/')[-1]
                if repo_name.endswith('.git'):
                    repo_name = repo_name[:-4]
                repo_name = repo_name.lower()
            else:
                # Process ZIP file
                if not zip_file.filename.lower().endswith('.zip'):
                    flash("Uploaded file must be a ZIP archive", "error")
                    return render_template('index.html')
                
                # Create temporary directory
                temp_dir = tempfile.mkdtemp()
                zip_filename = secure_filename(zip_file.filename)
                zip_path = os.path.join(temp_dir, zip_filename)
                zip_file.save(zip_path)
                
                # Extract ZIP file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Get repo name from ZIP filename
                repo_name = os.path.splitext(zip_filename)[0].lower()
                local_repo_path = temp_dir

            # Load configuration from session
            config_data = session.get('user_config', DEFAULT_CONFIG.copy())
           
            # Create documentation configuration
            config = DocumentationConfig(
                include_overview=config_data.get('include_overview', DEFAULT_CONFIG['include_overview']),
                include_executive_summary=config_data.get('include_executive_summary', DEFAULT_CONFIG['include_executive_summary']),
                include_tech_stack=config_data.get('include_tech_stack', DEFAULT_CONFIG['include_tech_stack']),
                include_code_structure=config_data.get('include_code_structure', DEFAULT_CONFIG['include_code_structure']),
                include_loc_analysis=config_data.get('include_loc_analysis', DEFAULT_CONFIG['include_loc_analysis']),
                include_loc_chart=config_data.get('include_loc_chart', DEFAULT_CONFIG['include_loc_chart']),
                include_complexity_analysis=config_data.get('include_complexity_analysis', DEFAULT_CONFIG['include_complexity_analysis']),
                include_complexity_charts=config_data.get('include_complexity_charts', DEFAULT_CONFIG['include_complexity_charts']),
                include_features=config_data.get('include_features', DEFAULT_CONFIG['include_features']),
                include_dependencies=config_data.get('include_dependencies', DEFAULT_CONFIG['include_dependencies']),
                include_issues=config_data.get('include_issues', DEFAULT_CONFIG['include_issues']),
                include_sql_objects=config_data.get('include_sql_objects', DEFAULT_CONFIG['include_sql_objects']),
                include_class_diagram=config_data.get('include_class_diagram', DEFAULT_CONFIG['include_class_diagram']),
                include_flow_diagram=config_data.get('include_flow_diagram', DEFAULT_CONFIG['include_flow_diagram']),
                include_er_diagram=config_data.get('include_er_diagram', DEFAULT_CONFIG['include_er_diagram']),
                include_reference_architecture=config_data.get('include_reference_architecture', DEFAULT_CONFIG['include_reference_architecture']),
                max_files_to_analyze=config_data.get('max_files_to_analyze', DEFAULT_CONFIG['max_files_to_analyze'])
            )
           
            # Initialize documentation generator
            doc_generator = GitHubDocGenerator(
                deployment_name=DEPLOYMENT_NAME,
                api_key=API_KEY,
                api_base=API_BASE,
                github_token=app.config['GITHUB_TOKEN'],
                output_dir=str(OUTPUT_DIR),
                config=config
            )
           
            if github_link:
                documentation = doc_generator.generate_documentation(github_link)
            else:
                documentation = doc_generator.generate_documentation_from_local(local_repo_path, repo_name)
            doc_metadata = []
            for doc_type in ['pdf', 'markdown', 'class_diagram', 'flow_diagram', 'reference_architecture', 'er_diagram']:
                doc_path = documentation.get(f'{doc_type}_path')
                if doc_path and os.path.exists(doc_path):
                    # Upload to blob storage
                    blob_url = upload_to_blob_storage(
                        file_path=doc_path,
                        user_id=current_user.id,
                        document_type=doc_type,
                        repo_name=repo_name
                    )
                    
                    # Index the document if it's a PDF or markdown
                    if doc_type == 'pdf':
                        try:
                            content = extract_text_from_pdf(doc_path)
                            index_document_chunks(
                                repo_name=repo_name,
                                document_name=os.path.basename(doc_path),
                                document_type=doc_type,
                                content=content,
                                blob_url=blob_url,
                                user_id=current_user.id
                            )
                        except Exception as e:
                            app.logger.error(f"Indexing failed for {doc_type}: {str(e)}")                    
                   
                    doc_metadata.append({
                        'type': doc_type,
                        'url': blob_url,
                        'name': os.path.basename(doc_path),
                        'repo_name': repo_name
                    })
           
            # Upload generated documents to Azure Blob Storage
            for doc_type in ['markdown', 'pdf', 'class_diagram', 'flow_diagram', 'reference_architecture','er_diagram']:
                doc_path = documentation.get(f'{doc_type}_path')
                if doc_path and os.path.exists(doc_path):
                    blob_url = upload_to_blob_storage(
                        file_path=doc_path,  # Correct parameter name
                        user_id=current_user.id,
                        document_type=doc_type,
                        repo_name=repo_name
                    )
                    doc_metadata.append({
                        'type': doc_type,
                        'url': blob_url,
                        'name': os.path.basename(doc_path),
                        'repo_name': repo_name
                    })
 
            if not documentation:
                flash("Failed to generate documentation", "error")
                return render_template('index.html')
           
            # Store results in session
            session['documentation'] = {
                'markdown_path': os.path.basename(documentation.get('markdown_path', '')),
                'pdf_path': os.path.basename(documentation.get('pdf_path', '')),
                'class_diagram_path': os.path.basename(documentation.get('class_diagram_path', '')),
                'flow_diagram_path': os.path.basename(documentation.get('flow_diagram_path', '')),
                'er_diagram_path': os.path.basename(documentation.get('er_diagram_path', '')),
                'reference_architecture_path': os.path.basename(documentation.get('reference_architecture_path', '')),
                'loc_chart_path': os.path.basename(documentation.get('loc_chart_path', '')),
                'repo_name': repo_name
            }
           
            # Store document metadata in database
            conn = get_db_connection()
            cursor = conn.cursor()
            for doc in doc_metadata:
                cursor.execute(
                    """INSERT INTO documents
                    (user_id, document_name, blob_url, document_type, repo_name)
                    VALUES (?, ?, ?, ?, ?)""",
                    (current_user.id, doc['name'], doc['url'], doc['type'], doc['repo_name'])
                )
            conn.commit()
            conn.close()

            return redirect(url_for('index'))
       
        except Exception as e:
            app.logger.error(f"Error in documentation generation: {str(e)}")
            flash(f"Error generating documentation: {str(e)}", "error")
            return render_template('index.html')
        
        finally:
            # Clean up temporary directory for ZIP files
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    return render_template('index.html')
@app.route('/documents')
@login_required
def documents():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
         SELECT TOP 5
                id,
                document_name AS name,  # This alias is critical
                document_type AS type, # This alias is critical
                created_at
            FROM documents
            WHERE user_id = ?
            ORDER BY created_at DESC
    """, (current_user.id,))
    documents = cursor.fetchall()
    conn.close()
    return render_template('documents.html', documents=documents)
 
@app.route('/result')
@login_required
def result():
    documentation = session.get('documentation')
    if not documentation:
        flash("No documentation results found", "error")
        return redirect(url_for('index'))
   
    # Debug the variable to check what's available
    print("Available documentation keys:", documentation.keys())
    print("Reference architecture path:", documentation.get('reference_architecture_path'))
   
    return render_template('result.html',
        markdown_path=documentation.get('markdown_path'),
        pdf_path=documentation.get('pdf_path'),
        class_diagram_path=documentation.get('class_diagram_path'),
        flow_diagram_path=documentation.get('flow_diagram_path'),
        er_diagram_path=documentation.get('er_diagram_path'),
        reference_architecture_path=documentation.get('reference_architecture_path'),
        loc_chart_path=documentation.get('loc_chart_path')
    )
 
@app.route('/user-documents')
@login_required
def user_documents():
    """List all repositories and documents for the current user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get all unique repositories for the current user
        cursor.execute("""
            SELECT 
                repo_name,
                COUNT(*) as doc_count
            FROM documents
            WHERE user_id = ?
            GROUP BY repo_name
            ORDER BY MAX(created_at) DESC
        """, (current_user.id,))
        
        repositories = []
        for row in cursor.fetchall():
            repositories.append({
                'repo_name': row.repo_name,
                'doc_count': row.doc_count
            })
        
        # Get recent documents for the dropdown
        cursor.execute("""
            SELECT TOP 5
                id,
                document_name as name,
                document_type as type
            FROM documents
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, (current_user.id,))
        
        recent_docs = []
        for row in cursor.fetchall():
            recent_docs.append({
                'id': row.id,
                'name': row.name,
                'type': row.type
            })
        
        return render_template('user_documents.html',
                             repositories=repositories,
                             recent_docs=recent_docs)
    
    except Exception as e:
        flash(f"Error retrieving documents: {str(e)}", "error")
        return redirect(url_for('index'))
    finally:
        conn.close()
       
 
 
# New route to delete a document
@app.route('/delete-document/<int:doc_id>', methods=['POST'])
@login_required
def delete_document(doc_id):
    """Delete a document from both database and blob storage"""
    conn = get_db_connection()
    cursor = conn.cursor()
   
    try:
        # Verify document belongs to current user
        cursor.execute("""
            SELECT document_name, document_type
            FROM documents
            WHERE id = ? AND user_id = ?
        """, (doc_id, current_user.id))
       
        doc = cursor.fetchone()
        if not doc:
            return jsonify({'status': 'error', 'message': 'Document not found'}), 404
       
        # Delete from blob storage
        blob_client = BlobServiceClient.from_connection_string(
            app.config['AZURE_STORAGE_CONNECTION_STRING']
        ).get_blob_client(
            container=app.config['AZURE_STORAGE_CONTAINER'],
            blob=f"{current_user.id}/{doc.document_type}/{doc.document_name}"
        )
        blob_client.delete_blob()
       
        # Delete from database
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
       
        return jsonify({'status': 'success', 'message': 'Document deleted'})
   
    except Exception as e:
        conn.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()
 
@app.route('/download/<filename>')
def download_file(filename):
    try:
        # Secure the filename
        filename = secure_filename(filename)
       
        # Ensure the file exists
        file_path = OUTPUT_DIR / filename
        if not file_path.exists():
            flash(f"File {filename} not found", "error")
            return render_template('index.html')
       
        return send_file(
            str(file_path),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        flash(f"Error downloading file: {str(e)}", "error")
        return render_template('index.html')
 
@app.route('/view/<filename>')
def view_file(filename):
    try:
        # Secure the filename
        filename = secure_filename(filename)
       
        # Ensure the file exists
        file_path = OUTPUT_DIR / filename
        if not file_path.exists():
            flash(f"File {filename} not found", "error")
            return render_template('index.html')
       
        # Read and convert markdown to HTML for viewing
        if filename.endswith('.md'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                html_content = markdown.markdown(content, extensions=['fenced_code', 'tables'])
                return render_template('markdown_view.html', content=html_content)
       
        return send_file(str(file_path))
    except Exception as e:
        flash(f"Error viewing file: {str(e)}", "error")
        return render_template('index.html')
 
# Configuration routes
@app.route('/configuration', methods=['GET'])
@login_required
def configuration():
    """Render the configuration page"""
    return render_template('configuration.html')
 
@app.route('/get-config', methods=['GET'])
@login_required
def get_config():
    """Get the current configuration from the session"""
    try:
        # If configuration is not in session, initialize with defaults
        if 'user_config' not in session:
            session['user_config'] = DEFAULT_CONFIG.copy()
           
            # Also update the config file with defaults to keep it in sync
            config_path = os.path.join(app.instance_path, 'config.json')
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
           
            with open(config_path, 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
       
        return jsonify(session['user_config'])
   
    except Exception as e:
        app.logger.error(f"Error loading configuration: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error loading configuration: {str(e)}"
        }), 500
 
@app.route('/save-config', methods=['POST'])
@login_required
def save_config():
    """Save configuration to both session and file"""
    try:
        # Get configuration data from request
        config_data = request.json
       
        # Ensure max_files_to_analyze is an integer if present
        if 'max_files_to_analyze' in config_data:
            try:
                config_data['max_files_to_analyze'] = int(config_data['max_files_to_analyze'])
                if config_data['max_files_to_analyze'] < 1:
                    config_data['max_files_to_analyze'] = 1
            except (ValueError, TypeError):
                config_data['max_files_to_analyze'] = 5
       
        # Save configuration to session
        session['user_config'] = config_data
       
        # Also save to file
        config_path = os.path.join(app.instance_path, 'config.json')
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
       
        return jsonify({
            'status': 'success',
            'message': 'Configuration saved successfully'
        })
   
    except Exception as e:
        app.logger.error(f"Error saving configuration: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error saving configuration: {str(e)}"
        }), 500
 
@app.route('/reset-config', methods=['POST'])
@login_required
def reset_config():
    """Reset configuration to default values"""
    # Reset session config
    session['user_config'] = DEFAULT_CONFIG.copy()
   
    # Also reset the file config
    config_path = os.path.join(app.instance_path, 'config.json')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
       
    return jsonify({
        'status': 'success',
        'message': 'Configuration reset to defaults'
    })
 
@app.route('/get-language-configs', methods=['GET'])
@login_required
def get_language_configs():
    """Get all available language configurations"""
    try:
        # Initialize doc generator to get default language configs
        doc_generator = GitHubDocGenerator(
            deployment_name=DEPLOYMENT_NAME,
            api_key=API_KEY,
            api_base=API_BASE,
            github_token=app.config['GITHUB_TOKEN'],
            output_dir=str(OUTPUT_DIR)
        )
       
        # Updated language list to include Python and RPG
        languages = ["COBOL", "Java", "Python", "VB.NET", "SQL", "C", "ABAP", "RPG"]
        configs = {}
       
        for lang in languages:
            configs[lang] = doc_generator._get_language_specific_config(lang)
       
        return jsonify({
            'status': 'success',
            'language_configs': configs
        })
   
    except Exception as e:
        app.logger.error(f"Error getting language configs: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error getting language configs: {str(e)}"
        }), 500
 
@app.route('/save-language-config', methods=['POST'])
@login_required
def save_language_config():
    """Save a specific language configuration"""
    try:
        data = request.json
        language = data.get('language')
        config = data.get('config')
       
        if not language or not config:
            return jsonify({
                'status': 'error',
                'message': 'Language and config are required'
            }), 400
       
        # Load existing config file or create new
        config_path = os.path.join(app.instance_path, 'language_configs.json')
        language_configs = {}
       
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                language_configs = json.load(f)
       
        # Update the specific language config
        language_configs[language] = config
       
        # Save back to file
        with open(config_path, 'w') as f:
            json.dump(language_configs, f, indent=2)
       
        return jsonify({
            'status': 'success',
            'message': f'{language} configuration saved successfully'
        })
   
    except Exception as e:
        app.logger.error(f"Error saving language config: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error saving language config: {str(e)}"
        }), 500

@app.route('/get-indexed-repos', methods=['GET'])
@login_required
def get_indexed_repos():
    try:
        service_name = app.config['AZURE_SEARCH_SERVICE_NAME']
        admin_key = app.config['AZURE_SEARCH_ADMIN_KEY']
        index_name = app.config['AZURE_SEARCH_INDEX_NAME']
        
        search_client = SearchClient(
            endpoint=f"https://{service_name}.search.windows.net",
            index_name=index_name,
            credential=AzureKeyCredential(admin_key)
        )
        
        # Get unique repositories for the current user
        results = search_client.search(
            search_text="*",
            filter=f"user_id eq '{current_user.id}'",
            facets=["repo_name"]
        )
        
        # Extract unique repo names from facets
        repos = set()
        if results.get_facets() and 'repo_name' in results.get_facets():
            for facet in results.get_facets()['repo_name']:
                repos.add(facet['value'])
        
        return jsonify({'repositories': list(repos)})
    except Exception as e:
        app.logger.error(f"Error getting indexed repos: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/delete-repo/<repo_name>', methods=['POST'])
@login_required
def delete_repo(repo_name):
    conn = get_db_connection()
    cursor = conn.cursor()
   
    try:
        # Verify repository exists and belongs to user
        cursor.execute("""
            SELECT COUNT(*) FROM documents
            WHERE user_id = ? AND repo_name = ?
        """, (current_user.id, repo_name))
        if cursor.fetchone()[0] == 0:
            flash("Repository not found", "danger")
            return redirect(url_for('user_documents'))
 
        # Get all documents in the repository
        cursor.execute("""
            SELECT blob_url FROM documents
            WHERE user_id = ? AND repo_name = ?
        """, (current_user.id, repo_name))
        documents = cursor.fetchall()
 
        # Delete all blobs from Azure storage
        blob_service_client = BlobServiceClient.from_connection_string(
            app.config['AZURE_STORAGE_CONNECTION_STRING']
        )
       
        deleted_blobs = 0
        for doc in documents:
            try:
                parsed_url = urlparse(doc[0])
                path = unquote(parsed_url.path).lstrip('/')  # Handle URL encoding
                parts = path.split('/')
               
                if len(parts) < 2:
                    continue  # Skip invalid URLs
                   
                container_name = parts[0]
                blob_name = '/'.join(parts[1:])
               
                blob_client = blob_service_client.get_blob_client(
                    container=container_name,
                    blob=blob_name
                )
                # Delete blob and its snapshots
                blob_client.delete_blob(delete_snapshots="include")
                deleted_blobs += 1
               
            except Exception as blob_error:
                app.logger.error(f"Error deleting blob {doc[0]}: {str(blob_error)}")
                continue
 
        # Delete all database records
        cursor.execute("""
            DELETE FROM documents
            WHERE user_id = ? AND repo_name = ?
        """, (current_user.id, repo_name))
        conn.commit()
 
        flash(f'Repository "{repo_name}" has been successfully deleted.', 'success')
        return redirect(url_for('user_documents'))
 
    except Exception as e:
        conn.rollback()
        app.logger.error(f"Repository deletion error: {str(e)}")
        flash(f"Error deleting repository: {str(e)}", "danger")
        return redirect(url_for('user_documents'))
    finally:
        conn.close()

@app.route('/generate-token')
def generate_token():
    return render_template('githubtoken.html')
 
# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    flash("The requested page was not found", "error")
    return render_template('index.html'), 404
 
@app.errorhandler(500)
def internal_server_error(e):
    flash("An internal server error occurred", "error")
    return render_template('index.html'), 500
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
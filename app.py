from flask import Flask, render_template, request, jsonify
import requests
import os
from werkzeug.utils import secure_filename

# Library imports with safety checks
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx
except ImportError:
    docx = None

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

app = Flask(__name__)

# --- CONFIGURATION ---
# Using environment variable for model, with a sensible default
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/api/chat")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b") 
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'md', 'pdf', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()
    filename = os.path.basename(filepath)
    if ext == 'pdf':
        if PdfReader is None: return f"[Error: Install pypdf: python3 -m pip install pypdf]"
        try:
            reader = PdfReader(filepath)
            return "".join([p.extract_text() + "\n" for p in reader.pages if p.extract_text()])
        except Exception as e: return f"[PDF Error {filename}: {str(e)}]"
    elif ext == 'docx':
        if docx is None: return f"[Error: Install python-docx: python3 -m pip install python-docx]"
        try:
            doc = docx.Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e: return f"[DOCX Error {filename}: {str(e)}]"
    else:
        try:
            with open(filepath, 'r', encoding='utf-8') as f: return f.read()
        except Exception as e: return f"[File Error {filename}: {str(e)}]"

def perform_web_search(query):
    if DDGS is None:
        return "[Error: Install duckduckgo_search: python3 -m pip install duckduckgo_search]"
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=3)]
            return "\n\n".join(results) if results else "No web results found."
    except Exception as e:
        return f"[Web Search Error: {str(e)}]"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use pdf, docx, txt, md"}), 400
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return jsonify({"message": f"Uploaded {filename}!"})

@app.route('/files', methods=['GET'])
def list_files():
    return jsonify({"files": os.listdir(app.config['UPLOAD_FOLDER'])})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages', [])
    if not messages: return jsonify({"error": "No messages"}), 400
    
    user_query = messages[-1]['content']
    
    # 1. Web Search Logic
    web_triggers = ["search", "internet", "web", "current", "latest", "news", "today"]
    web_context = ""
    if any(t in user_query.lower() for t in web_triggers):
        web_context = f"\n--- LIVE WEB RESULTS ---\n{perform_web_search(user_query)}\n--- END WEB ---\n"

    # 2. Document Logic
    doc_context = ""
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    if files:
        doc_context = "\n--- UPLOADED DOCUMENTS ---\n"
        for f in files:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f)
            doc_context += f"\nFile: {f}\nContent:\n{extract_text(filepath)}\n"
        doc_context += "\n--- END DOCUMENTS ---\n"

    # Inject combined context into the last message
    modified_messages = []
    for msg in messages:
        if msg == messages[-1] and msg['role'] == 'user':
            content = f"Context:\n{doc_context}\n{web_context}\n\nUser Question: {msg['content']}"
            modified_messages.append({"role": "user", "content": content})
        else:
            modified_messages.append(msg)

    try:
        payload = {
            "model": MODEL_NAME,
            "messages": modified_messages,
            "stream": False
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        
        if response.status_code == 404:
            return jsonify({"error": f"Model '{MODEL_NAME}' not found. Please run 'ollama pull {MODEL_NAME}' in your terminal."}), 404
            
        if response.status_code != 200:
            return jsonify({"error": f"Ollama returned status {response.status_code}: {response.text}"}), response.status_code
            
        res_json = response.json()
        answer = res_json.get('message', {}).get('content')
        
        if not answer:
            return jsonify({"error": "Ollama returned an empty response."}), 500
            
        return jsonify({"success": True, "response": answer})
        
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Ollama server is not reachable. Ensure Ollama is running on port 11434."}), 503
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

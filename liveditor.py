
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import json
import os
from datetime import datetime
import re

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)  # More secure secret key generation

# Configuration
SAVES_DIR = 'saved_diagrams'
if not os.path.exists(SAVES_DIR):
    os.makedirs(SAVES_DIR)

class DiagramValidator:
    @staticmethod
    def validate_syntax(code):
        """Validates the syntax of a Mermaid diagram code.
        
        Supports multiple diagram types including:
        - Flowcharts
        - Sequence diagrams
        - Class diagrams
        - State diagrams
        - Entity Relationship diagrams
        - User Journey diagrams
        - Gantt charts
        - Pie charts
        - And other supported Mermaid diagram types
        """
        # Check if the code is empty
        if not code.strip():
            return False, "Diagram code cannot be empty"
        
        # Check for basic syntax patterns
        valid_starts = [
                'flowchart', 'sequenceDiagram', 'classDiagram', 'stateDiagram',
                'erDiagram', 'journey', 'gantt', 'pie', 'gitGraph', 'graph',
                'mindmap', 'timeline', 'zenuml', 'quadrantChart', 'requirementDiagram',
                'C4Context', 'C4Container', 'C4Component', 'C4Dynamic', 'C4Deployment',  # Updated C4 syntax
                'sankey-beta'  # New diagram type
            ]
        
        # Extract first word/token to determine diagram type
        first_line = code.strip().split('\n')[0].strip()
        first_token = first_line.split(' ')[0] if ' ' in first_line else first_line
        
        # Check if the diagram starts with a valid diagram type declaration
        if not any(first_token.startswith(start) for start in valid_starts):
            return False, f"Invalid diagram type. Must start with one of: {', '.join(valid_starts)}"
        
        # Check for balanced braces and brackets
        if code.count('{') != code.count('}'):
            return False, "Unbalanced braces in diagram"
        
        if code.count('[') != code.count(']'):
            return False, "Unbalanced brackets in diagram"
        
        if code.count('(') != code.count(')'):
            return False, "Unbalanced parentheses in diagram"
        
        return True, "Diagram syntax appears valid"

@app.route('/')
def index():
    return render_template('liveditor.html')

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
            
        if data.get('mermaid_version', '10.9.0') < '10.9.0':
            if any(keyword in diagram_code for keyword in ['sankey-beta', 'blockDiagram']):
                return jsonify({
                    'status': 'error',
                    'message': 'This diagram requires Mermaid 10.9.0 or newer'
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
            'mermaid_version': data.get('mermaid_version', '10.9.0'),
            'diagram_type': data.get('diagram_type', 'unknown'),
            'features_used': detect_features(data.get('code', '')),
            'version': '3.0'  # For future compatibility
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

def detect_features(code):
            features = []
            if 'sankey-beta' in code:
                features.append('sankey')
            if 'C4Context' in code:
                features.append('c4')
            if 'mindmap' in code:
                features.append('mindmap')
            return features

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
                        'mermaid_version': diagram.get('mermaid_version', '10.8.0'),
                        'diagram_type': diagram.get('diagram_type', 'unknown'),
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

@app.route('/mermaid-versions', methods=['GET'])
def get_mermaid_versions():
    """Return available Mermaid.js versions and their supported features"""
    versions = {
        '10.9.0': {
            'supported_diagrams': [
                'flowchart', 'sequenceDiagram', 'classDiagram', 'stateDiagram',
                'erDiagram', 'journey', 'gantt', 'pie', 'gitGraph', 'graph',
                'mindmap', 'timeline', 'zenuml', 'quadrantChart', 'requirementDiagram',
                'C4Context', 'C4Container', 'C4Component', 'C4Dynamic', 'C4Deployment',
                'sankey-beta', 'blockDiagram'
            ],
            'cdn_url': 'https://cdn.jsdelivr.net/npm/mermaid@10.9.0/dist/mermaid.min.js',
            'is_latest': True
        },
        '10.9.0': {
            'supported_diagrams': [
                # ... keep previous version's capabilities
            ]
        }
    }
    return jsonify({'status': 'success', 'versions': versions})

@app.route('/check-compatibility', methods=['POST'])
def check_compatibility():
    try:
        data = request.json
        diagram_code = data.get('code', '')
        
        # Detect required Mermaid version
        required_version = '10.8.0'
        if any(keyword in diagram_code for keyword in ['sankey-beta', 'C4Dynamic', 'blockDiagram']):
            required_version = '10.9.0'
        
        return jsonify({
            'status': 'success',
            'required_version': required_version,
            'recommended_cdn': f'https://cdn.jsdelivr.net/npm/mermaid@{required_version}/dist/mermaid.min.js'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    


if __name__ == '__main__':
    app.run(debug=True)
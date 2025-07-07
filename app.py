import os
import uuid
import io  # <-- NEW: Import the io module for in-memory handling
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename

from comparator_engine import compare_pdfs, create_diff_pdf

# --- Flask App Setup ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Main Application Routes ---

@app.route('/')
def index():
    """Renders the main upload page (index.html)."""
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    """Handles file upload, runs comparison, and returns the diff as JSON."""
    if 'pdf1' not in request.files or 'pdf2' not in request.files:
        return jsonify({'status': 'error', 'message': 'Both PDF files are required.'}), 400

    file1 = request.files['pdf1']
    file2 = request.files['pdf2']

    if file1.filename == '' or file2.filename == '':
        return jsonify({'status': 'error', 'message': 'Please select two files.'}), 400

    filename1 = secure_filename(file1.filename)
    filename2 = secure_filename(file2.filename)
    path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
    file1.save(path1)
    file2.save(path2)

    diff_result = []
    try:
        diff_result = compare_pdfs(path1, path2)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'An error occurred during comparison: {str(e)}'}), 500
    finally:
        if os.path.exists(path1):
            os.remove(path1)
        if os.path.exists(path2):
            os.remove(path2)

    return jsonify({'status': 'success', 'diff': diff_result})

# --- PDF Generation Route (Corrected) ---

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    """
    Receives diff data, generates a PDF in memory, and returns it for download.
    """
    data = request.json
    if not data or 'diff' not in data:
        return jsonify({'status': 'error', 'message': 'Invalid diff data provided.'}), 400

    diff_result = data['diff']
    
    # Define a temporary filename. We still need to write it to disk first.
    output_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"diff_{uuid.uuid4()}.pdf")
    
    buffer = None
    try:
        # Step 1: Create the PDF file on disk as before.
        create_diff_pdf(diff_result, output_filename)
        
        # Step 2: Open the file and read its binary content into a buffer.
        # The 'with' statement ensures the file handle is closed immediately after reading.
        with open(output_filename, 'rb') as f:
            buffer = io.BytesIO(f.read())
        
        # This is crucial: move the buffer's "cursor" to the beginning.
        buffer.seek(0)
        
    except Exception as e:
        print(f"PDF Generation Error: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to generate PDF: {str(e)}'}), 500
    finally:
        # Step 3: Now that the file is read into memory (or if an error occurred),
        # we can safely delete the physical file from the disk. This will not error.
        if os.path.exists(output_filename):
            os.remove(output_filename)

    if buffer is None:
        # This case handles if the buffer was never created due to an error.
        return jsonify({'status': 'error', 'message': 'PDF buffer could not be created.'}), 500

    # Step 4: Send the file from the in-memory buffer.
    return send_file(
        buffer,
        as_attachment=True,
        download_name='comparison_report.pdf',
        mimetype='application/pdf'  # Explicitly set the MIME type
    )

if __name__ == '__main__':
    app.run(debug=True)
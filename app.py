from flask import Flask, request, send_file, render_template
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import os
import tempfile

app = Flask(__name__)

# Route 1: Show the upload page
@app.route('/')
def index():
    return render_template('index.html')

# Route 2: Process the uploaded file
@app.route('/process', methods=['POST'])
def process_pdf():
    if 'pdf_file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['pdf_file']
    if file.filename == '':
        return "No selected file", 400

    # Create temporary files to handle the processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_in, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_out:
        
        # Save the uploaded file temporarily
        file.save(temp_in.name)
        
        # --- OUR PDF PROCESSING LOGIC ---
        doc = fitz.open(temp_in.name)
        page = doc.load_page(0)
        
        zoom = 300 / 72 
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        image_rgb_original = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        image_bgr = cv2.cvtColor(image_rgb_original, cv2.COLOR_RGB2BGR)
        
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        
        # The color net we fine-tuned
        lower_brown_orange = np.array([3, 60, 120])
        upper_brown_orange = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower_brown_orange, upper_brown_orange)
        
        image_bgr[mask > 0] = (255, 255, 255)
        
        final_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(final_rgb)
        
        # Save the processed image to the temporary output file
        pil_image.save(temp_out.name, "PDF", resolution=300.0)
        doc.close()

    # Send the cleaned file back to the user, then clean up the temp files
    response = send_file(temp_out.name, as_attachment=True, download_name="Cleaned_Plan.pdf")
    os.remove(temp_in.name)
    os.remove(temp_out.name)
    
    return response

if __name__ == '__main__':
    app.run(debug=True)

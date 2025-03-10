from flask import Flask, request, jsonify, send_from_directory
from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
import time
import uuid
import pytesseract
from PIL import Image
import easyocr
import io
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize OCR engines
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Initialize EasyOCR
easy_reader = easyocr.Reader(['en'])

# Tesseract config
tesseract_config = r'--oem 3 --psm 11'

@app.route('/api/paddle-ocr', methods=['POST'])
def paddle_ocr_endpoint():
    """
    Endpoint for OCR processing using PaddleOCR algorithm.
    Accepts an image file and returns detected text with bounding boxes.
    """
    # Check if image file is included in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'Empty file name'}), 400
    
    # Generate a unique filename to avoid collisions
    filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save the uploaded file
    file.save(filepath)
    
    try:
        # Start timing
        start_time = time.time()
        
        # Process with PaddleOCR
        result = paddle_ocr.ocr(filepath, cls=True)
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Process results
        if not result or len(result) == 0 or len(result[0]) == 0:
            return jsonify({
                'error': 'No text detected in the image',
                'elapsed_time': elapsed_time
            }), 404
        
        # Extract detection results
        detections = []
        for line in result[0]:
            box = line[0]  # Coordinates of the bounding box
            text = line[1][0]  # Detected text
            confidence = float(line[1][1])  # Confidence score
            
            detections.append({
                'box_coordinates': box,
                'text': text,
                'confidence': confidence
            })
        
        # Optional: Generate an annotated image
        image = cv2.imread(filepath)
        annotated_image_path = None
        
        if image is not None:
            for detection in detections:
                box = detection['box_coordinates']
                text = detection['text']
                confidence = detection['confidence']
                
                # Convert to proper format for OpenCV
                box_points = [(int(pt[0]), int(pt[1])) for pt in box]
                box_np = np.array(box_points)
                
                # Draw the bounding box
                cv2.polylines(image, [box_np], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Add text label
                x, y = box_points[0]
                cv2.putText(image, f"{text} ({confidence:.2f})", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Save the annotated image
            annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + filename)
            cv2.imwrite(annotated_image_path, image)
            annotated_image_path = 'annotated_' + filename
        
        # Prepare the response
        response = {
            'success': True,
            'elapsed_time': elapsed_time,
            'detections': detections,
            'annotated_image': annotated_image_path,
            'total_items_detected': len(detections),
            'engine': 'paddle'
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up - you may want to keep files for debugging or remove this in production
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass

@app.route('/api/tesseract-ocr', methods=['POST'])
def tesseract_ocr_endpoint():
    """
    Endpoint for OCR processing using Tesseract algorithm.
    Accepts an image file and returns detected text with bounding boxes.
    """
    # Check if image file is included in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'Empty file name'}), 400
    
    # Generate a unique filename to avoid collisions
    filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save the uploaded file
    file.save(filepath)
    
    try:
        # Start timing
        start_time = time.time()
        
        # Process with Tesseract
        image = Image.open(filepath)
        
        # Get data including bounding boxes (page segmentation mode 11 = sparse text)
        data = pytesseract.image_to_data(image, config=tesseract_config, output_type=pytesseract.Output.DICT)
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Process results
        detections = []
        n_boxes = len(data['text'])
        
        # Original image for annotations
        cv_image = cv2.imread(filepath)
        
        for i in range(n_boxes):
            # Filter out empty text and low confidence detections
            if int(data['conf'][i]) > 60 and data['text'][i].strip() != '':
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                # Box coordinates in format similar to PaddleOCR for consistency
                box = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                text = data['text'][i]
                confidence = int(data['conf'][i]) / 100.0
                
                detections.append({
                    'box_coordinates': box,
                    'text': text,
                    'confidence': confidence
                })
                
                # Draw rectangle and text
                cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(cv_image, f"{text} ({confidence:.2f})", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Save the annotated image
        annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + filename)
        cv2.imwrite(annotated_image_path, cv_image)
        
        # Prepare the response
        response = {
            'success': True,
            'elapsed_time': elapsed_time,
            'detections': detections,
            'annotated_image': 'annotated_' + filename,
            'total_items_detected': len(detections),
            'engine': 'tesseract'
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass

@app.route('/api/easyocr', methods=['POST'])
def easyocr_endpoint():
    """
    Endpoint for OCR processing using EasyOCR algorithm.
    Accepts an image file and returns detected text with bounding boxes.
    """
    # Check if image file is included in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'Empty file name'}), 400
    
    # Generate a unique filename to avoid collisions
    filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save the uploaded file
    file.save(filepath)
    
    try:
        # Start timing
        start_time = time.time()
        
        # Process with EasyOCR
        results = easy_reader.readtext(filepath)
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Original image for annotations
        image = cv2.imread(filepath)
        
        # Process results
        detections = []
        for (box, text, confidence) in results:
            # Convert box to format similar to PaddleOCR
            box_coords = [[int(box[0][0]), int(box[0][1])], 
                          [int(box[1][0]), int(box[1][1])], 
                          [int(box[2][0]), int(box[2][1])], 
                          [int(box[3][0]), int(box[3][1])]]
            
            detections.append({
                'box_coordinates': box_coords,
                'text': text,
                'confidence': confidence
            })
            
            # Draw the bounding box
            points = np.array([[int(p[0]), int(p[1])] for p in box])
            cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Add text label
            cv2.putText(image, f"{text} ({confidence:.2f})", 
                       (int(box[0][0]), int(box[0][1] - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Save the annotated image
        annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + filename)
        cv2.imwrite(annotated_image_path, image)
        
        # Prepare the response
        response = {
            'success': True,
            'elapsed_time': elapsed_time,
            'detections': detections,
            'annotated_image': 'annotated_' + filename,
            'total_items_detected': len(detections),
            'engine': 'easyocr'
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass

@app.route('/api/ocr-compare', methods=['POST'])
def ocr_compare_endpoint():
    """
    Endpoint that compares all OCR engines and returns consolidated results.
    Useful for comparing performance and accuracy across different algorithms.
    """
    # Check if image file is included in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'Empty file name'}), 400
    
    # Generate a unique filename to avoid collisions
    filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save the uploaded file
    file.save(filepath)
    
    try:
        results = {}
        
        # Process with PaddleOCR
        paddle_start = time.time()
        paddle_result = paddle_ocr.ocr(filepath, cls=True)
        paddle_time = time.time() - paddle_start
        
        paddle_detections = []
        if paddle_result and len(paddle_result) > 0 and len(paddle_result[0]) > 0:
            for line in paddle_result[0]:
                paddle_detections.append({
                    'box_coordinates': line[0],
                    'text': line[1][0],
                    'confidence': float(line[1][1])
                })
        
        results['paddle'] = {
            'elapsed_time': paddle_time,
            'detections': paddle_detections,
            'total_items_detected': len(paddle_detections)
        }
        
        # Process with Tesseract
        tesseract_start = time.time()
        image = Image.open(filepath)
        data = pytesseract.image_to_data(image, config=tesseract_config, output_type=pytesseract.Output.DICT)
        tesseract_time = time.time() - tesseract_start
        
        tesseract_detections = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 60 and data['text'][i].strip() != '':
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                box = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                text = data['text'][i]
                confidence = int(data['conf'][i]) / 100.0
                
                tesseract_detections.append({
                    'box_coordinates': box,
                    'text': text,
                    'confidence': confidence
                })
        
        results['tesseract'] = {
            'elapsed_time': tesseract_time,
            'detections': tesseract_detections,
            'total_items_detected': len(tesseract_detections)
        }
        
        # Process with EasyOCR
        easyocr_start = time.time()
        easyocr_results = easy_reader.readtext(filepath)
        easyocr_time = time.time() - easyocr_start
        
        easyocr_detections = []
        for (box, text, confidence) in easyocr_results:
            box_coords = [[int(box[0][0]), int(box[0][1])], 
                          [int(box[1][0]), int(box[1][1])], 
                          [int(box[2][0]), int(box[2][1])], 
                          [int(box[3][0]), int(box[3][1])]]
            
            easyocr_detections.append({
                'box_coordinates': box_coords,
                'text': text,
                'confidence': confidence
            })
        
        results['easyocr'] = {
            'elapsed_time': easyocr_time,
            'detections': easyocr_detections,
            'total_items_detected': len(easyocr_detections)
        }
        
        # Create a combined visualization
        original_img = cv2.imread(filepath)
        h, w = original_img.shape[:2]
        
        # Create a canvas for all visualizations side by side
        canvas = np.zeros((h, w*3, 3), dtype=np.uint8)
        
        # Draw PaddleOCR results
        paddle_img = original_img.copy()
        for det in paddle_detections:
            box = det['box_coordinates']
            text = det['text']
            confidence = det['confidence']
            
            box_points = [(int(pt[0]), int(pt[1])) for pt in box]
            box_np = np.array(box_points)
            
            cv2.polylines(paddle_img, [box_np], isClosed=True, color=(0, 255, 0), thickness=2)
            x, y = box_points[0]
            cv2.putText(paddle_img, f"{text[:10]}...", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw Tesseract results
        tesseract_img = original_img.copy()
        for det in tesseract_detections:
            box = det['box_coordinates']
            text = det['text']
            confidence = det['confidence']
            
            x, y = int(box[0][0]), int(box[0][1])
            w = int(box[2][0]) - x
            h = int(box[2][1]) - y
            
            cv2.rectangle(tesseract_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(tesseract_img, f"{text[:10]}...", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw EasyOCR results
        easyocr_img = original_img.copy()
        for det in easyocr_detections:
            box = det['box_coordinates']
            text = det['text']
            confidence = det['confidence']
            
            points = np.array([[int(p[0]), int(p[1])] for p in box])
            cv2.polylines(easyocr_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            x, y = int(box[0][0]), int(box[0][1])
            cv2.putText(easyocr_img, f"{text[:10]}...", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Place images in canvas
        canvas[:, 0:w] = paddle_img
        canvas[:, w:2*w] = tesseract_img
        canvas[:, 2*w:] = easyocr_img
        
        # Add labels
        cv2.putText(canvas, "PaddleOCR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "Tesseract", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "EasyOCR", (2*w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save comparison image
        comparison_path = os.path.join(app.config['UPLOAD_FOLDER'], 'comparison_' + filename)
        cv2.imwrite(comparison_path, canvas)
        
        # Add comparison image path to response
        response = {
            'success': True,
            'comparison_image': 'comparison_' + filename,
            'results': results
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass

@app.route('/uploads/<filename>', methods=['GET'])
def get_uploaded_file(filename):
    """
    Endpoint to retrieve annotated images.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
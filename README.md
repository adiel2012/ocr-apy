 
# Multi-Engine OCR API

A comprehensive OCR (Optical Character Recognition) REST API that leverages multiple OCR engines to provide robust text detection capabilities. This application combines PaddleOCR, Tesseract, and EasyOCR in a single API, allowing users to compare performance across different OCR technologies.

## 📋 Features

- **Multiple OCR Engines**:
  - PaddleOCR: State-of-the-art OCR framework from Baidu
  - Tesseract OCR: Google's open-source OCR engine
  - EasyOCR: Python library for text detection with deep learning
  
- **Comparison Endpoint**: Run all engines simultaneously and compare their results

- **Visualization**: Annotated images showing detected text with bounding boxes

- **Performance Metrics**: Processing time and confidence scores for each detection

- **Docker Support**: Easily deployable via Docker and Docker Compose

## 🚀 Getting Started

### Prerequisites

#### Option 1: Docker (Recommended)

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

#### Option 2: Local Installation

- Python 3.9+
- Tesseract OCR engine installed on your system
- Various Python packages (see requirements.txt)

### Installation

#### Using Docker (Recommended)

1. **Run the dockerize.bat script to create all necessary files**:
   ```bash
   dockerize.bat
   ```

2. **Build and start the container**:
   ```bash
   docker-compose up --build
   ```

   This will start the API server on port 5000.

3. **Or run in detached mode**:
   ```bash
   docker-compose up -d
   ```

#### Manual Installation (Not recommended on Windows)

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR**:
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`
   - Windows: Download from [UB-Mannheim's GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

4. **Run the application**:
   ```bash
   python app.py
   ```

## 🔍 API Endpoints

### 1. PaddleOCR Endpoint

```
POST /api/paddle-ocr
```

Process an image with PaddleOCR, known for its high accuracy with complex layouts.

**Request**:
- Form data with an image file: `image=@your_image.jpg`

**Response**:
```json
{
  "success": true,
  "elapsed_time": 1.25,
  "detections": [
    {
      "box_coordinates": [[100, 100], [200, 100], [200, 150], [100, 150]],
      "text": "SAMPLE TEXT",
      "confidence": 0.98
    }
  ],
  "annotated_image": "annotated_abc123.jpg",
  "total_items_detected": 1,
  "engine": "paddle"
}
```

### 2. Tesseract OCR Endpoint

```
POST /api/tesseract-ocr
```

Process an image with Tesseract OCR, reliable for clean printed text.

**Request**:
- Form data with an image file: `image=@your_image.jpg`

**Response**: Similar structure to PaddleOCR endpoint.

### 3. EasyOCR Endpoint

```
POST /api/easyocr
```

Process an image with EasyOCR, which performs well on natural scene text.

**Request**:
- Form data with an image file: `image=@your_image.jpg`

**Response**: Similar structure to PaddleOCR endpoint.

### 4. Comparison Endpoint

```
POST /api/ocr-compare
```

Process an image with all three OCR engines and compare results.

**Request**:
- Form data with an image file: `image=@your_image.jpg`

**Response**:
```json
{
  "success": true,
  "comparison_image": "comparison_abc123.jpg",
  "results": {
    "paddle": {
      "elapsed_time": 1.25,
      "detections": [...],
      "total_items_detected": 8
    },
    "tesseract": {
      "elapsed_time": 0.75,
      "detections": [...],
      "total_items_detected": 7
    },
    "easyocr": {
      "elapsed_time": 1.5,
      "detections": [...],
      "total_items_detected": 6
    }
  }
}
```

### 5. Image Retrieval Endpoint

```
GET /uploads/<filename>
```

Retrieve annotated or comparison images generated by the OCR processors.

## 📝 Usage Examples

### Using cURL

```bash
# PaddleOCR
curl -X POST -F "image=@sample.jpg" http://localhost:5000/api/paddle-ocr

# Tesseract OCR
curl -X POST -F "image=@sample.jpg" http://localhost:5000/api/tesseract-ocr

# EasyOCR
curl -X POST -F "image=@sample.jpg" http://localhost:5000/api/easyocr

# Compare all engines
curl -X POST -F "image=@sample.jpg" http://localhost:5000/api/ocr-compare
```

### Using the Test Script

A Python test script is included to easily test the API:

```bash
python test_ocr_api.py sample.jpg paddle-ocr
python test_ocr_api.py sample.jpg tesseract-ocr
python test_ocr_api.py sample.jpg easyocr
python test_ocr_api.py sample.jpg ocr-compare
```

## 🔧 Configuration

The OCR engines can be configured by modifying the parameters in `app.py`:

```python
# PaddleOCR configuration
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Tesseract configuration
tesseract_config = r'--oem 3 --psm 11'

# EasyOCR configuration
easy_reader = easyocr.Reader(['en'])
```

## 🧪 Performance Considerations

- **PaddleOCR**: Higher accuracy for complex layouts, but slower processing
- **Tesseract**: Fast for clean printed text, struggles with complex layouts
- **EasyOCR**: Good for natural scene text, medium speed

## 📦 Project Structure

```
multi-engine-ocr-api/
├── app.py               # Main Flask application
├── Dockerfile           # Docker image configuration
├── docker-compose.yml   # Docker Compose configuration
├── requirements.txt     # Python dependencies
├── uploads/             # Directory for storing processed images
├── test_ocr_api.py      # Test script for the API
└── README.md            # This documentation
```

## 🔒 Security Notes

- The API doesn't implement authentication or rate limiting
- Temporary files are removed after processing for security
- For production use, consider adding authentication and HTTPS

## 🛠️ Troubleshooting

### Docker Issues

1. **Container fails to start**:
   ```bash
   docker-compose logs
   ```

2. **Permission issues with uploads directory**:
   ```bash
   chmod 777 uploads  # On Linux/Mac
   ```

3. **Access container shell for debugging**:
   ```bash
   docker-compose exec ocr-api bash
   ```

### Common OCR Issues

1. **No text detected**:
   - Try a different OCR engine
   - Ensure the image has sufficient resolution
   - Check if text is oriented properly

2. **Poor recognition accuracy**:
   - Improve image quality/contrast
   - Use preprocessing (not implemented in this API)
   - Try a different OCR engine

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Flask](https://flask.palletsprojects.com/)
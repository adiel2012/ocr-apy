# OCR API Docker Setup

This document explains how to run the multi-engine OCR API using Docker and Docker Compose.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Getting Started

1. Clone or create the project with all the necessary files:
   - `app.py` - The Flask application
   - `Dockerfile` - Instructions for building the Docker image
   - `docker-compose.yml` - Configuration for Docker Compose
   - `requirements.txt` - Python dependencies

2. Build and start the container:
   ```bash
   docker-compose up --build
   ```

   This will:
   - Build the Docker image with all required dependencies
   - Start the OCR API container
   - Map port 5000 from the container to port 5000 on your host
   - Mount the `./uploads` directory for persistent storage

3. To run in detached mode (in the background):
   ```bash
   docker-compose up -d
   ```

4. To stop the container:
   ```bash
   docker-compose down
   ```

## API Endpoints

Once the container is running, you can access the following endpoints:

- `http://localhost:5000/api/paddle-ocr` - OCR using PaddleOCR
- `http://localhost:5000/api/tesseract-ocr` - OCR using Tesseract
- `http://localhost:5000/api/easyocr` - OCR using EasyOCR
- `http://localhost:5000/api/ocr-compare` - Compare results from all engines

## Example Usage

Using curl to send an image for OCR processing:

```bash
curl -X POST -F "image=@sample.jpg" http://localhost:5000/api/paddle-ocr
```

Or using a tool like Postman to send a POST request with an image file.

## Troubleshooting

- If you encounter permission issues with the `uploads` directory, check the permissions on your host machine:
  ```bash
  chmod 777 uploads
  ```

- If the container fails to start, check the logs:
  ```bash
  docker-compose logs
  ```

- To access a shell inside the running container for debugging:
  ```bash
  docker-compose exec ocr-api bash
  ```

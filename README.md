# Face Recognition API

### FastAPI · DeepFace · Computer Vision · Docker · Google Cloud Run

A production-oriented **face verification API** built with **FastAPI and DeepFace**. The application accepts two face images, generates facial embeddings, compares them using **cosine similarity**, and returns a verification decision through both a web interface and REST API.

The project demonstrates how a computer-vision model can be transformed from Python experimentation into a **containerized, cloud-deployable machine-learning service**.

---

## 🚀 Demo

The application provides two ways to provide images:

* 📷 Capture an image directly using the browser camera
* 📤 Upload two existing face images

The system then:

```text
Image 1 ──┐
          ├──> Face Processing ──> DeepFace Embeddings
Image 2 ──┘                              │
                                         ▼
                                  Cosine Similarity
                                         │
                                         ▼
                                  Threshold Decision
                                    ↙           ↘
                                VERIFIED     NOT VERIFIED
```

---

## ✨ Features

* Face verification using **DeepFace**
* Facial embedding generation
* Cosine similarity-based comparison
* Browser-based camera capture
* Image upload support
* Interactive verification results
* Similarity score and threshold display
* REST API using FastAPI
* Automatic API documentation with Swagger/OpenAPI
* Health-check endpoint
* Docker containerization
* Google Cloud Run deployment support
* Configurable model, detector backend, and similarity threshold

---

## 🧠 How Face Verification Works

The application follows a simplified face-verification pipeline:

### 1. Image Input

Two images are provided:

```text
Image A → Reference / ID image
Image B → Verification / Selfie image
```

### 2. Face Processing

DeepFace processes the images and extracts facial representations.

### 3. Embedding Generation

Each face is converted into a numerical feature vector:

```text
Face Image
    ↓
DeepFace
    ↓
Face Embedding
[0.12, -0.34, 0.81, ...]
```

The default Facenet configuration produces a **128-dimensional embedding** in this project.

### 4. Similarity Calculation

The two embeddings are compared using cosine similarity:

```text
similarity = cosine(embedding_1, embedding_2)
```

A higher similarity indicates that the two embeddings are more similar.

### 5. Verification Decision

The similarity is compared against a configurable threshold:

```text
similarity >= threshold
        ↓
     VERIFIED
```

Otherwise:

```text
similarity < threshold
        ↓
   NOT VERIFIED
```

> **Important:** The default `0.70` threshold is an educational/demo operating point. It is **not a production biometric threshold**. A real identity-verification system requires threshold calibration using representative validation data and explicit FAR/FRR targets.

---

# 🏗️ Architecture

```text
                    ┌──────────────────────┐
                    │      Web Browser     │
                    │                      │
                    │ Camera / Image Upload│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      FastAPI         │
                    │                      │
                    │ REST API + Web UI    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      DeepFace        │
                    │                      │
                    │ Face Detection       │
                    │ Embedding Generation│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Similarity Engine    │
                    │                      │
                    │ Cosine Similarity    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Verification Result  │
                    │                      │
                    │ Verified / Rejected  │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Docker Container  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Google Cloud Run   │
                    └──────────────────────┘
```

---

# 🛠️ Tech Stack

| Technology              | Purpose                              |
| ----------------------- | ------------------------------------ |
| **Python**              | Core programming language            |
| **FastAPI**             | REST API and web application         |
| **DeepFace**            | Face representation and verification |
| **OpenCV**              | Image processing                     |
| **NumPy**               | Numerical computation                |
| **HTML/CSS/JavaScript** | Browser interface and camera capture |
| **Docker**              | Containerization                     |
| **Google Cloud Run**    | Cloud deployment                     |
| **Uvicorn**             | ASGI application server              |

---

# 📁 Project Structure

```text
face_recognition_fastapi/
│
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
├── .dockerignore           # Docker build exclusions
├── README.md
│
├── templates/
│   └── index.html          # Web interface
│
└── static/
    └── style.css           # UI styling
```

---

# ⚙️ Configuration

The application supports the following environment variables:

```env
FACE_MODEL=Facenet
DETECTOR_BACKEND=opencv
SIMILARITY_THRESHOLD=0.70
```

### `FACE_MODEL`

Specifies the DeepFace model used for generating facial representations.

Default:

```text
Facenet
```

### `DETECTOR_BACKEND`

Specifies the face detector backend.

Default:

```text
opencv
```

### `SIMILARITY_THRESHOLD`

Controls the verification decision.

Default:

```text
0.70
```

For example:

```text
Similarity = 0.84
Threshold = 0.70

0.84 >= 0.70
       ↓
   VERIFIED
```

---

# 💻 Running Locally

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Start the application

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

---

# 📚 API Documentation

FastAPI automatically generates interactive API documentation.

### Swagger UI

```text
http://127.0.0.1:8000/docs
```

### ReDoc

```text
http://127.0.0.1:8000/redoc
```

---

# 🔌 REST API

## Verify Two Faces

### Endpoint

```http
POST /api/verify
```

### Request

Send two image files using multipart form data:

```bash
curl -X POST http://localhost:8000/api/verify \
  -F "image1=@person1.jpg" \
  -F "image2=@person2.jpg"
```

### Example Response

```json
{
  "verified": true,
  "similarity": 0.84,
  "threshold": 0.7,
  "model": "Facenet",
  "embedding_dimension": 128
}
```

---

# ❤️ Health Check

### Endpoint

```http
GET /health
```

Example:

```bash
curl http://localhost:8000/health
```

This endpoint can be used by container orchestration and cloud platforms to determine whether the application is running.

---

# 🐳 Docker

## Build the image

```bash
docker build -t face-recognition-api .
```

## Run the container

```bash
docker run --rm -p 8080:8080 face-recognition-api
```

Open:

```text
http://localhost:8080
```

API documentation:

```text
http://localhost:8080/docs
```

---

# ☁️ Deploy to Google Cloud Run

This application is containerized and can be deployed to Google Cloud Run.

## Authenticate

```bash
gcloud auth login
```

## Select your project

```bash
gcloud config set project PROJECT_ID
```

## Deploy from source

```bash
gcloud run deploy face-recognition-api \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated
```

After deployment, Google Cloud Run provides a public service URL.

Example:

```text
https://face-recognition-api-xxxxx.asia-south1.run.app
```

---

# 📦 Artifact Registry Deployment

For an explicit container-image workflow:

### Create a Docker repository

```bash
gcloud artifacts repositories create face-repo \
  --repository-format=docker \
  --location=asia-south1
```

### Build and push the image

```bash
gcloud builds submit \
  --tag asia-south1-docker.pkg.dev/PROJECT_ID/face-repo/face-recognition-api
```

### Deploy

```bash
gcloud run deploy face-recognition-api \
  --image asia-south1-docker.pkg.dev/PROJECT_ID/face-repo/face-recognition-api \
  --region asia-south1 \
  --allow-unauthenticated
```

---

# 🔐 Security & Production Considerations

This project is designed primarily as a **learning and portfolio implementation**.

A production-grade biometric verification system would require additional safeguards, including:

* HTTPS-only communication
* Authentication and authorization
* Rate limiting
* Secure image handling
* Input validation
* Request-size limits
* Temporary-file cleanup
* Encryption at rest and in transit
* Privacy-preserving data retention policies
* Liveness / presentation-attack detection
* Threshold calibration
* FAR/FRR evaluation
* Monitoring and logging
* Model-performance monitoring
* Protection against image injection and replay attacks

The current implementation should therefore **not be treated as a production KYC or identity-verification system**.

---

# 📊 Evaluation Metrics

For a real face-verification system, accuracy alone is not enough.

Important metrics include:

### False Acceptance Rate — FAR

The percentage of impostor attempts incorrectly accepted.

```text
Impostor → Accepted
```

Lower FAR generally means stronger security.

### False Rejection Rate — FRR

The percentage of genuine users incorrectly rejected.

```text
Genuine User → Rejected
```

Lower FRR generally means better user experience.

### Equal Error Rate — EER

The operating point where:

```text
FAR = FRR
```

Threshold selection should therefore be based on the desired security/UX operating point rather than arbitrarily choosing a similarity value.

---

# 🚀 Future Improvements

Potential extensions to this project include:

* [ ] Add passive liveness detection
* [ ] Add active liveness challenges
* [ ] Add Presentation Attack Detection
* [ ] Add deepfake detection
* [ ] Add face alignment
* [ ] Add quality checks for blur and illumination
* [ ] Add authentication to API endpoints
* [ ] Add rate limiting
* [ ] Add automated tests
* [ ] Add CI/CD with GitHub Actions
* [ ] Add structured logging
* [ ] Add Prometheus/Grafana monitoring
* [ ] Add model benchmarking
* [ ] Calibrate threshold using validation data
* [ ] Add 1:N face identification
* [ ] Add FAISS/vector database for large-scale search
* [ ] Optimize inference with ONNX/TensorRT
* [ ] Deploy GPU-backed inference where appropriate

---

# 🎯 What This Project Demonstrates

This project goes beyond simply calling a face-recognition library.

It demonstrates the complete path from:

```text
Machine Learning
      ↓
Computer Vision
      ↓
Face Embeddings
      ↓
Similarity Matching
      ↓
REST API
      ↓
FastAPI Application
      ↓
Docker
      ↓
Cloud Deployment
```

It therefore serves as a practical example of **ML Engineering + Computer Vision + Backend Development + Cloud Deployment**.

---

# 📌 Interview Talking Points

If discussing this project in an interview, be prepared to explain:

### Computer Vision

* How face detection works
* What a face embedding represents
* Why embeddings are used instead of raw pixels
* Cosine similarity vs Euclidean distance
* Face alignment
* Face recognition vs face verification
* 1:1 vs 1:N matching

### Machine Learning

* Threshold selection
* FAR vs FRR
* EER
* False positives vs false negatives
* Dataset leakage
* Model evaluation

### Backend

* FastAPI
* REST APIs
* Multipart image uploads
* API validation
* Health checks
* ASGI/Uvicorn

### DevOps

* Docker images vs containers
* Dockerfile
* Container ports
* Cloud Run
* Artifact Registry
* Stateless services
* Horizontal scaling

### Production ML

* Model latency
* Model optimization
* Quantization
* ONNX
* Monitoring
* Data drift
* Concept drift

---

# ⚠️ Disclaimer

This repository is intended for **educational and portfolio purposes**.

It should not be used as-is for high-stakes identity verification, financial KYC, surveillance, or authentication systems without appropriate security, privacy, biometric evaluation, liveness detection, threshold calibration, compliance, and production hardening.

---

## 👨‍💻 Author

**Subrat Jain**

Computer Science / AI & ML

If you found this project useful, consider giving the repository a ⭐.

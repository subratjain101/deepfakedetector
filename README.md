# Face Recognition API — FastAPI + DeepFace + Docker + Cloud Run

This converts the earlier DeepFace face-verification logic into a web application.

## Features

- Upload two face images.
- Capture Image 1 directly from the browser camera.
- Generate DeepFace embeddings.
- Compare embeddings with cosine similarity.
- Display same-person/different-person result, similarity, threshold and embedding dimension.
- JSON API at `POST /api/verify`.
- Health endpoint at `GET /health`.
- Dockerized for Google Cloud Run.

## Local run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`.

## Docker

```bash
docker build -t face-recognition-api .
docker run --rm -p 8080:8080 face-recognition-api
```

Open `http://127.0.0.1:8080`.

## Environment variables

```text
FACE_MODEL=Facenet
DETECTOR_BACKEND=opencv
SIMILARITY_THRESHOLD=0.70
```

The threshold is an educational/default operating point, not a production biometric threshold. It should be calibrated against representative validation data for the selected model and security/UX target.

## Google Cloud Run

```bash
gcloud auth login
gcloud config set project PROJECT_ID

gcloud run deploy face-recognition-api --source . --region REGION --allow-unauthenticated
```

Cloud Run can build and deploy directly from source. The included Dockerfile can also be built and pushed to Artifact Registry if you want an explicit image-based workflow.

For an image-based deployment:

```bash
gcloud artifacts repositories create face-repo --repository-format=docker --location=REGION
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/face-repo/face-recognition-api
gcloud run deploy face-recognition-api \
  --image REGION-docker.pkg.dev/PROJECT_ID/face-repo/face-recognition-api \
  --region REGION \
  --allow-unauthenticated
```

## API example

```bash
curl -X POST http://localhost:8080/api/verify \
  -F "image1=@person1.jpg" \
  -F "image2=@person2.jpg"
```

Example response:

```json
{
  "verified": true,
  "similarity": 0.84,
  "threshold": 0.7,
  "model": "Facenet",
  "embedding_dimension": 128
}
```

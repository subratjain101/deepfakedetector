import base64
import os
from typing import Optional

import cv2
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Face Recognition API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MODEL_NAME = os.getenv("FACE_MODEL", "Facenet")
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "opencv")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))


def bytes_to_image(data: bytes) -> np.ndarray:
    """Convert uploaded image bytes into an OpenCV BGR image."""
    if not data:
        raise ValueError("Empty image")

    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image file")

    return image


def get_embedding(image: np.ndarray) -> list[float]:
    """Generate a face embedding using DeepFace."""
    result = DeepFace.represent(
        img_path=image,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
        align=True,
    )

    return result[0]["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_vec = np.asarray(a, dtype=np.float32)
    b_vec = np.asarray(b, dtype=np.float32)

    denominator = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)
    if denominator == 0:
        raise ValueError("Cannot compare zero-length embeddings")

    return float(np.dot(a_vec, b_vec) / denominator)


def image_to_data_url(image: np.ndarray) -> str:
    """Convert an OpenCV image into a browser-displayable data URL."""
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Could not encode image")

    encoded_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_b64}"


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": None,
            "error": None,
            "threshold": SIMILARITY_THRESHOLD,
        },
    )


@app.post("/api/verify")
async def verify_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    """Compare two uploaded face images."""
    try:
        image1_bytes = await image1.read()
        image2_bytes = await image2.read()

        first = bytes_to_image(image1_bytes)
        second = bytes_to_image(image2_bytes)

        embedding1 = get_embedding(first)
        embedding2 = get_embedding(second)
        similarity = cosine_similarity(embedding1, embedding2)

        verified = similarity >= SIMILARITY_THRESHOLD

        return {
            "verified": verified,
            "similarity": round(similarity, 4),
            "threshold": SIMILARITY_THRESHOLD,
            "model": MODEL_NAME,
            "embedding_dimension": len(embedding1),
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # DeepFace can raise detector/model errors for images without a usable face.
        raise HTTPException(status_code=422, detail=f"Face verification failed: {exc}") from exc


@app.post("/api/verify-ui", response_class=HTMLResponse)
async def verify_ui(
    request: Request,
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    """Browser form endpoint that renders the result page."""
    try:
        image1_bytes = await image1.read()
        image2_bytes = await image2.read()

        first = bytes_to_image(image1_bytes)
        second = bytes_to_image(image2_bytes)

        embedding1 = get_embedding(first)
        embedding2 = get_embedding(second)
        similarity = cosine_similarity(embedding1, embedding2)
        verified = similarity >= SIMILARITY_THRESHOLD

        result = {
            "verified": verified,
            "similarity": round(similarity, 4),
            "threshold": SIMILARITY_THRESHOLD,
            "model": MODEL_NAME,
            "embedding_dimension": len(embedding1),
            "image1": image_to_data_url(first),
            "image2": image_to_data_url(second),
        }

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"result": result, "error": None, "threshold": SIMILARITY_THRESHOLD},
        )
    except ValueError as exc:
        error = str(exc)
    except Exception as exc:
        error = f"Face verification failed: {exc}"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": None, "error": error, "threshold": SIMILARITY_THRESHOLD},
        status_code=422,
    )


@app.get("/health")
def health():
    return {"status": "ok"}

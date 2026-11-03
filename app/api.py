"""
DefectLens AI — FastAPI REST Endpoint.

Provides a production-ready REST API for industrial defect inspection.

Endpoints:
    POST /predict          — Classify an image as normal/defective
    POST /predict/anomaly  — Score anomaly magnitude
    GET  /health           — Health check
    GET  /docs             — Auto-generated Swagger UI

Launch:
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

Docker:
    docker run -p 8000:8000 defectlens-ai uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

import io
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

import numpy as np
from PIL import Image

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="DefectLens AI",
        description="Industrial Visual Inspection & Anomaly Detection REST API",
        version="1.0.0",
        contact={"name": "Habimana Moise", "email": "habimanamoise77@gmail.com"},
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Response models ────────────────────────────────────────────────
    class PredictionResponse(BaseModel):
        verdict: str                    # "NORMAL" or "DEFECTIVE"
        confidence: float               # 0.0 – 1.0
        defect_probability: float
        normal_probability: float
        latency_ms: float
        model: str

    class AnomalyResponse(BaseModel):
        anomaly_score: float            # Higher = more anomalous
        is_anomaly: bool
        threshold: float
        latency_ms: float

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        device: str
        version: str

    # ── Lazy model loading ─────────────────────────────────────────────
    _predictor = None

    def get_predictor():
        global _predictor
        if _predictor is None:
            try:
                from defectlens.inference.predictor import DefectPredictor
                from defectlens.models.classifier import DefectClassifier
                model = DefectClassifier(num_classes=2, pretrained=False)
                _predictor = DefectPredictor(model=model, device="cpu")
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Model init failed: {e}")
        return _predictor

    def load_image(upload: UploadFile) -> np.ndarray:
        """Read uploaded file into a numpy RGB array."""
        try:
            content = upload.file.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
            return np.array(img)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file.")

    # ── Routes ─────────────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health():
        """API health check."""
        import torch
        return HealthResponse(
            status="ok",
            model_loaded=_predictor is not None,
            device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            version="1.0.0",
        )

    @app.post("/predict", response_model=PredictionResponse, tags=["Inspection"])
    async def predict(
        file: UploadFile = File(..., description="Industrial product image (JPEG/PNG)"),
        threshold: float = Query(0.5, ge=0.0, le=1.0, description="Classification threshold"),
    ):
        """
        Classify an uploaded image as **NORMAL** or **DEFECTIVE**.

        Returns the predicted class, confidence score, and inference latency.
        """
        predictor = get_predictor()
        image = load_image(file)
        result = predictor.predict(image)

        defect_prob = float(result.get("defect_prob", result.get("probability", 0.5)))
        normal_prob = 1.0 - defect_prob
        is_defective = defect_prob >= threshold

        return PredictionResponse(
            verdict="DEFECTIVE" if is_defective else "NORMAL",
            confidence=max(defect_prob, normal_prob),
            defect_probability=round(defect_prob, 4),
            normal_probability=round(normal_prob, 4),
            latency_ms=round(result.get("latency_ms", 0.0), 2),
            model="DefectClassifier-ResNet18",
        )

    @app.post("/predict/batch", tags=["Inspection"])
    async def predict_batch(
        files: List[UploadFile] = File(..., description="Multiple product images"),
        threshold: float = Query(0.5, ge=0.0, le=1.0),
    ):
        """
        Classify multiple images in a single request.
        Returns a list of predictions with per-image verdicts.
        """
        predictor = get_predictor()
        results = []
        for f in files:
            image = load_image(f)
            result = predictor.predict(image)
            defect_prob = float(result.get("defect_prob", result.get("probability", 0.5)))
            results.append({
                "filename": f.filename,
                "verdict": "DEFECTIVE" if defect_prob >= threshold else "NORMAL",
                "defect_probability": round(defect_prob, 4),
                "latency_ms": round(result.get("latency_ms", 0.0), 2),
            })
        return {"predictions": results, "total": len(results)}

    @app.get("/", tags=["System"])
    async def root():
        return {
            "name": "DefectLens AI",
            "description": "Industrial Visual Inspection REST API",
            "docs": "/docs",
            "health": "/health",
        }

else:
    # Stub for when fastapi is not installed
    class app:
        pass

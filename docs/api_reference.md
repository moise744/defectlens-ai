# DefectLens AI — API Reference

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## Endpoints

### `GET /health`

Health check. Returns service status and dependency availability.

**Response:**
```json
{
  "status": "ok",
  "service": "defectlens-ai",
  "fastapi": "True",
  "opencv": "True",
  "torch": "True",
  "defectlens_modules": "True"
}
```

---

### `POST /inspect`

Run defect inspection on an uploaded image.

**Request:** `multipart/form-data`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | ✅ | Image file (JPEG/PNG) |
| `mode` | string | — | `classifier`, `anomaly`, or `segmentor` (default: `classifier`) |
| `threshold` | float | — | Anomaly/mask threshold 0.0–1.0 (default: `0.5`) |

**Response:**
```json
{
  "filename": "bottle_test.jpg",
  "mode": "classifier",
  "verdict": "defective",
  "confidence": 0.923,
  "anomaly_score": null,
  "defect_pixel_fraction": null,
  "inference_time_ms": 45.2
}
```

---

### `POST /inspect/full`

Run all three inspection modes on a single image.

**Request:** `multipart/form-data`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | ✅ | Image file (JPEG/PNG) |

**Response:**
```json
{
  "filename": "surface.jpg",
  "classifier": {"verdict": "defective", "confidence": 0.91},
  "anomaly": {"score": 0.78, "verdict": "defective"},
  "segmentor": {"defect_pixel_fraction": 0.034, "verdict": "defective"}
}
```

---

### `POST /benchmark`

Measure inference latency for a given mode.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | string | `classifier` | Inspection mode |
| `n_runs` | int | `20` | Number of inference runs |

**Response:**
```json
{
  "mode": "classifier",
  "n_runs": 20,
  "mean_ms": 47.3,
  "std_ms": 3.1,
  "min_ms": 43.2,
  "max_ms": 58.7,
  "fps": 21.1
}
```

---

### `GET /models`

List available inspection models and their status.

**Response:**
```json
{
  "models": [
    {"name": "classifier", "type": "ResNet-18", "status": "ready"},
    {"name": "anomaly", "type": "PatchCore", "status": "ready"},
    {"name": "segmentor", "type": "U-Net", "status": "ready"}
  ]
}
```

---

## Error Responses

| HTTP Code | Meaning |
|-----------|---------|
| `400` | Invalid or unreadable image file |
| `422` | Invalid query parameter value |
| `503` | Required module or model not available |

---

## Python Client Example

```python
import requests

with open("surface.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/inspect",
        files={"file": f},
        params={"mode": "classifier"},
    )

result = response.json()
print(result["verdict"], result["confidence"])
```

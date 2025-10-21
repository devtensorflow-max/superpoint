# app.py
from io import BytesIO
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from PIL import Image
from matcher import SuperPointMatcher, MODEL_NAME
import requests
from pydantic import BaseModel, HttpUrl

app = FastAPI(title="SuperPoint Match API", version="1.0")

# Load the model once at startup
@app.on_event("startup")
def _load_model():
    app.state.matcher = SuperPointMatcher()  # uses env SUPERPOINT_LOCAL_DIR if set

@app.get("/health")
def health():
    return {"status": "ok", "model_source": getattr(app.state.matcher, "source", "unknown")}

@app.post("/match")
async def match_images(
    image1: UploadFile = File(..., description="First image file"),
    image2: UploadFile = File(..., description="Second image file"),

    # Optional tuning parameters (all have sane defaults)
    upscale: float = Query(1.5, ge=1.0, le=3.0),
    ratio: float = Query(0.9, ge=0.5, le=0.99),
    mutual_check: bool = Query(False),
    ransac_reproj: float = Query(3.0, ge=0.1, le=10.0),
    ransac_iters: int = Query(5000, ge=100, le=100000),
    ransac_conf: float = Query(0.999, ge=0.5, le=0.9999),
    geom_model: str = Query("homography", pattern="^(homography|affine)$"),

    score_min: float = Query(0.30, ge=0.0, le=1.0),
    min_inliers: int = Query(12, ge=0),
    min_inlier_ratio: float = Query(0.15, ge=0.0, le=1.0),
    max_mean_inlier_desc_dist: float = Query(1.20, ge=0.0, le=2.0),
    min_mean_score_product: float = Query(0.12, ge=0.0, le=1.0),
    adaptive_score_factor: float = Query(0.50, ge=0.0, le=2.0),
    fraction_high_score: float = Query(0.50, ge=0.0, le=1.0),
    high_score_gate: float = Query(0.15, ge=0.0, le=1.0),
):
    def _valid_img(content_type: Optional[str]) -> bool:
        return content_type and any(
            content_type.lower().startswith(ct)
            for ct in ("image/", "application/octet-stream")
        )

    if not _valid_img(image1.content_type) or not _valid_img(image2.content_type):
        raise HTTPException(status_code=415, detail="Both files must be images")

    try:
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()
        img1 = Image.open(BytesIO(img1_bytes)).convert("RGB")
        img2 = Image.open(BytesIO(img2_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode images: {e}")

    matcher = app.state.matcher
    result = matcher.evaluate(
        img1, img2,
        upscale=upscale,
        ratio=ratio,
        mutual_check=mutual_check,
        ransac_reproj=ransac_reproj,
        ransac_iters=ransac_iters,
        ransac_conf=ransac_conf,
        geom_model=geom_model,
        score_min=score_min,
        min_inliers=min_inliers,
        min_inlier_ratio=min_inlier_ratio,
        max_mean_inlier_desc_dist=max_mean_inlier_desc_dist,
        min_mean_score_product=min_mean_score_product,
        adaptive_score_factor=adaptive_score_factor,
        fraction_high_score=fraction_high_score,
        high_score_gate=high_score_gate,
    )
    return JSONResponse(result)



# ✅ NEW schema for /match-urls
class UrlPair(BaseModel):
    url1: HttpUrl
    url2: HttpUrl


@app.post("/match-urls")
def match_from_urls(
    payload: UrlPair,
    upscale: float = Query(1.5, ge=1.0, le=3.0),
    ratio: float = Query(0.9, ge=0.5, le=0.99),
    mutual_check: bool = Query(False),
    ransac_reproj: float = Query(3.0, ge=0.1, le=10.0),
    ransac_iters: int = Query(5000, ge=100, le=100000),
    ransac_conf: float = Query(0.999, ge=0.5, le=0.9999),
    geom_model: str = Query("homography", pattern="^(homography|affine)$"),

    score_min: float = Query(0.30, ge=0.0, le=1.0),
    min_inliers: int = Query(12, ge=0),
    min_inlier_ratio: float = Query(0.15, ge=0.0, le=1.0),
    max_mean_inlier_desc_dist: float = Query(1.20, ge=0.0, le=2.0),
    min_mean_score_product: float = Query(0.12, ge=0.0, le=1.0),
    adaptive_score_factor: float = Query(0.50, ge=0.0, le=2.0),
    fraction_high_score: float = Query(0.50, ge=0.0, le=1.0),
    high_score_gate: float = Query(0.15, ge=0.0, le=1.0),
):
    """Match two images provided as URLs"""
    try:
        resp1 = requests.get(str(payload.url1), stream=True, timeout=10)
        resp2 = requests.get(str(payload.url2), stream=True, timeout=10)
        resp1.raise_for_status()
        resp2.raise_for_status()
        img1 = Image.open(BytesIO(resp1.content)).convert("RGB")
        img2 = Image.open(BytesIO(resp2.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch images: {e}")

    matcher = app.state.matcher
    result = matcher.evaluate(
        img1, img2,
        upscale=upscale,
        ratio=ratio,
        mutual_check=mutual_check,
        ransac_reproj=ransac_reproj,
        ransac_iters=ransac_iters,
        ransac_conf=ransac_conf,
        geom_model=geom_model,
        score_min=score_min,
        min_inliers=min_inliers,
        min_inlier_ratio=min_inlier_ratio,
        max_mean_inlier_desc_dist=max_mean_inlier_desc_dist,
        min_mean_score_product=min_mean_score_product,
        adaptive_score_factor=adaptive_score_factor,
        fraction_high_score=fraction_high_score,
        high_score_gate=high_score_gate,
    )
    return JSONResponse(result)



@app.get("/")
def root():
    return {
        "message": "SuperPoint Match API",
        "post_to": "/match",
        "docs": "/docs",
        "model": MODEL_NAME,
    }

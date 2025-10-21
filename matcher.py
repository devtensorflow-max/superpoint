# matcher.py
import os
import json
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SuperPointForKeypointDetection

# -------------------------
# Defaults (tweakable via API query params)
# -------------------------
UPSCALE = 1.5
RATIO = 0.9
MUTUAL_CHECK = False
RANSAC_REPROJ = 3.0
RANSAC_ITERS = 5000
RANSAC_CONF = 0.999

MIN_INLIERS = 12
MIN_INLIER_RATIO = 0.15
MAX_MEAN_INLIER_DESC_DIST = 1.20
SCORE_MIN = 0.30
MIN_MEAN_SCORE_PRODUCT = 0.12

ADAPTIVE_SCORE_FACTOR = 0.50
FRACTION_HIGH_SCORE = 0.50
HIGH_SCORE_GATE = 0.15

MODEL_NAME = "magic-leap-community/superpoint"
GEOM_MODEL = "homography"  # or "affine"

# -------------------------
# Helpers
# -------------------------
def pil_upscale(img: Image.Image, scale: float):
    if scale == 1.0:
        return img
    w, h = img.size
    return img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

def _looks_like_model_dir(path: Optional[str]) -> bool:
    if not (path and os.path.isdir(path)):
        return False
    names = set(os.listdir(path))
    has_config = any(n.startswith("config") and n.endswith(".json") for n in names)
    has_weights = any(
        n.endswith(".bin") or n.endswith(".safetensors") or n == "pytorch_model.bin"
    for n in names)
    return has_config and has_weights

def load_superpoint(local_dir: Optional[str]=None, model_id: str=MODEL_NAME):
    # Prefer explicit local dir, then env var, then local HF cache, then online
    candidates = []
    if local_dir:
        candidates.append(local_dir)
    env_dir = os.getenv("SUPERPOINT_LOCAL_DIR")
    if env_dir:
        candidates.append(env_dir)

    for d in candidates:
        if _looks_like_model_dir(d):
            try:
                proc = AutoImageProcessor.from_pretrained(d, local_files_only=True)
                mdl  = SuperPointForKeypointDetection.from_pretrained(d, local_files_only=True)
                print(f"[Model] Loaded SuperPoint from local directory: {os.path.abspath(d)}")
                return proc, mdl, f"local:{os.path.abspath(d)}"
            except Exception as e:
                print(f"[Model] Found local directory but failed to load from '{d}': {e}")

    try:
        proc = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
        mdl  = SuperPointForKeypointDetection.from_pretrained(model_id, local_files_only=True)
        print(f"[Model] Loaded SuperPoint from local HF cache for '{model_id}'.")
        return proc, mdl, f"cache:{model_id}"
    except Exception:
        print(f"[Model] Not in local cache for '{model_id}', will download from Hugging Face…")

    proc = AutoImageProcessor.from_pretrained(model_id)
    mdl  = SuperPointForKeypointDetection.from_pretrained(model_id)
    print(f"[Model] Downloaded SuperPoint from Hugging Face repo '{model_id}'.")
    return proc, mdl, f"huggingface:{model_id}"

def superpoint_features(images_pil: List[Image.Image], processor, model):
    with torch.inference_mode():
        inputs = processor(images_pil, return_tensors="pt")
        outputs = model(**inputs)
        image_sizes = [(im.height, im.width) for im in images_pil]  # (H,W)
        proc = processor.post_process_keypoint_detection(outputs, image_sizes)

    out = []
    for d in proc:
        kpts = d["keypoints"].cpu().numpy().astype(np.float32)      # (N,2) x,y
        desc = d["descriptors"].cpu().numpy().astype(np.float32)    # (N,256)
        sc   = d["scores"].cpu().numpy().astype(np.float32)         # (N,)

        eps = 1e-8
        nrm = np.linalg.norm(desc, axis=1, keepdims=True) + eps
        desc = desc / nrm

        out.append(dict(keypoints=kpts, descriptors=desc, scores=sc))
    return out

def filter_by_score(kpts, desc, scores, min_score=SCORE_MIN):
    if len(scores) == 0:
        return kpts, desc, scores, np.arange(0)
    keep = np.where(scores >= min_score)[0]
    return kpts[keep], desc[keep], scores[keep], keep

def match_descriptors(desc1, desc2, ratio=0.8, mutual=False):
    if desc1.size == 0 or desc2.size == 0:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn12 = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for pair in knn12:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if mutual and len(good) > 0:
        knn21 = bf.knnMatch(desc2, desc1, k=1)
        best21 = {m.queryIdx: m.trainIdx for [m] in knn21 if len([m]) == 1}
        good_mutual = []
        for m in good:
            if best21.get(m.trainIdx, -1) == m.queryIdx:
                good_mutual.append(m)
        good = good_mutual

    return good

def estimate_geometry(kpts1, kpts2, matches, model="homography",
                      ransac_reproj=3.0, max_iters=5000, conf=0.999):
    if len(matches) < 3:
        return None, None
    pts1 = np.float32([kpts1[m.queryIdx] for m in matches])  # (N,2)
    pts2 = np.float32([kpts2[m.trainIdx] for m in matches])  # (N,2)

    if model == "homography":
        M, mask = cv2.findHomography(
            pts1, pts2, cv2.RANSAC,
            ransacReprojThreshold=ransac_reproj,
            maxIters=max_iters,
            confidence=conf
        )
    else:
        M, mask = cv2.estimateAffine2D(
            pts1, pts2, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_reproj,
            maxIters=max_iters,
            confidence=conf,
            refineIters=10
        )
    return M, mask

def _adaptive_floor_from_means(scores1_f, scores2_f):
    m1 = float(np.mean(scores1_f)) if len(scores1_f) else 0.0
    m2 = float(np.mean(scores2_f)) if len(scores2_f) else 0.0
    return ADAPTIVE_SCORE_FACTOR * (m1 * m2)

def decision_from_all_signals(
    kpts1, kpts2, scores1, scores2, desc1, desc2, matches, inlier_mask,
    thresholds: Dict
) -> Tuple[bool, Dict]:
    num_good = len(matches)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    inlier_ratio = (inliers / max(num_good, 1))

    inlier_dists = []
    score_products = []
    if inlier_mask is not None:
        for m, keep in zip(matches, inlier_mask.ravel().tolist()):
            if keep:
                inlier_dists.append(m.distance)
                score_products.append(float(scores1[m.queryIdx]) * float(scores2[m.trainIdx]))

    mean_inlier_dist = float(np.mean(inlier_dists)) if len(inlier_dists) else float("inf")
    mean_score_prod  = float(np.mean(score_products)) if len(score_products) else 0.0

    adaptive_floor = thresholds["ADAPTIVE_SCORE_FACTOR"] * (
        float(np.mean(scores1)) * float(np.mean(scores2))
    )
    score_mean_ok = (mean_score_prod >= max(thresholds["MIN_MEAN_SCORE_PRODUCT"], adaptive_floor))
    frac_high = float(np.mean(np.array(score_products) >= thresholds["HIGH_SCORE_GATE"])) if score_products else 0.0
    score_frac_ok = (frac_high >= thresholds["FRACTION_HIGH_SCORE"])

    is_match = (
        (inliers >= thresholds["MIN_INLIERS"]) and
        (inlier_ratio >= thresholds["MIN_INLIER_RATIO"]) and
        (mean_inlier_dist <= thresholds["MAX_MEAN_INLIER_DESC_DIST"]) and
        (score_mean_ok or score_frac_ok)
    )

    diag = {
        "good_matches": num_good,
        "inliers": inliers,
        "inlier_ratio": round(inlier_ratio, 4),
        "mean_inlier_desc_dist": round(mean_inlier_dist if np.isfinite(mean_inlier_dist) else 9.99, 4),
        "mean_score_product": round(mean_score_prod, 4),
        "adaptive_score_floor": round(adaptive_floor, 4),
        "frac_high_score_inliers": round(frac_high, 4),
        "thresholds": thresholds
    }
    return is_match, diag

class SuperPointMatcher:
    def __init__(self, local_model_dir: Optional[str]=None, model_name: str=MODEL_NAME):
        self.processor, self.model, self.source = load_superpoint(local_model_dir, model_name)
        self.model.eval()
        print(f"[Model] Using source -> {self.source}")

    def evaluate(
        self,
        img1: Image.Image,
        img2: Image.Image,
        upscale: float = UPSCALE,
        ratio: float = RATIO,
        mutual_check: bool = MUTUAL_CHECK,
        ransac_reproj: float = RANSAC_REPROJ,
        ransac_iters: int = RANSAC_ITERS,
        ransac_conf: float = RANSAC_CONF,
        geom_model: str = GEOM_MODEL,
        score_min: float = SCORE_MIN,
        min_inliers: int = MIN_INLIERS,
        min_inlier_ratio: float = MIN_INLIER_RATIO,
        max_mean_inlier_desc_dist: float = MAX_MEAN_INLIER_DESC_DIST,
        min_mean_score_product: float = MIN_MEAN_SCORE_PRODUCT,
        adaptive_score_factor: float = ADAPTIVE_SCORE_FACTOR,
        fraction_high_score: float = FRACTION_HIGH_SCORE,
        high_score_gate: float = HIGH_SCORE_GATE
    ) -> Dict:
        # Prepare thresholds dict for output (keep keys stable)
        thresholds = {
            "MIN_INLIERS": int(min_inliers),
            "MIN_INLIER_RATIO": float(min_inlier_ratio),
            "MAX_MEAN_INLIER_DESC_DIST": float(max_mean_inlier_desc_dist),
            "MIN_MEAN_SCORE_PRODUCT": float(min_mean_score_product),
            "ADAPTIVE_SCORE_FACTOR": float(adaptive_score_factor),
            "HIGH_SCORE_GATE": float(high_score_gate),
            "FRACTION_HIGH_SCORE": float(fraction_high_score),
            "SCORE_MIN": float(score_min)
        }

        # Optional: upscale
        img1_u = pil_upscale(img1.convert("RGB"), upscale)
        img2_u = pil_upscale(img2.convert("RGB"), upscale)

        # Extract features
        feats = superpoint_features([img1_u, img2_u], self.processor, self.model)
        kpts1, desc1, scores1 = feats[0]["keypoints"], feats[0]["descriptors"], feats[0]["scores"]
        kpts2, desc2, scores2 = feats[1]["keypoints"], feats[1]["descriptors"], feats[1]["scores"]

        # Score pre-filter
        kpts1_f, desc1_f, scores1_f, idx1 = filter_by_score(kpts1, desc1, scores1, score_min)
        kpts2_f, desc2_f, scores2_f, idx2 = filter_by_score(kpts2, desc2, scores2, score_min)

        if desc1_f.size == 0 or desc2_f.size == 0:
            diag = {
                "good_matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "mean_inlier_desc_dist": 9.99,
                "mean_score_product": 0.0,
                "adaptive_score_floor": round(_adaptive_floor_from_means(scores1_f, scores2_f), 4),
                "frac_high_score_inliers": 0.0,
                "reason": "No descriptors after score filtering",
                "thresholds": thresholds
            }
            return {"match": "FALSE", "diagnostics": diag}

        # Descriptor matching
        good = match_descriptors(desc1_f, desc2_f, ratio=ratio, mutual=mutual_check)

        if len(good) < 4:
            diag = {
                "good_matches": len(good),
                "inliers": 0,
                "inlier_ratio": 0.0,
                "mean_inlier_desc_dist": 9.99,
                "mean_score_product": 0.0,
                "adaptive_score_floor": round(_adaptive_floor_from_means(scores1_f, scores2_f), 4),
                "frac_high_score_inliers": 0.0,
                "reason": "Not enough good matches for geometry (need >= 4)",
                "thresholds": thresholds
            }
            return {"match": "FALSE", "diagnostics": diag}

        # Geometry
        M, inlier_mask = estimate_geometry(
            kpts1_f, kpts2_f, good, model=geom_model,
            ransac_reproj=ransac_reproj, max_iters=ransac_iters, conf=ransac_conf
        )

        if M is None or inlier_mask is None:
            diag = {
                "good_matches": len(good),
                "inliers": 0,
                "inlier_ratio": 0.0,
                "mean_inlier_desc_dist": 9.99,
                "mean_score_product": 0.0,
                "adaptive_score_floor": round(_adaptive_floor_from_means(scores1_f, scores2_f), 4),
                "frac_high_score_inliers": 0.0,
                "reason": "Geometry estimation failed (RANSAC returned None)",
                "thresholds": thresholds
            }
            return {"match": "FALSE", "diagnostics": diag}

        # Decision
        is_match, diag = decision_from_all_signals(
            kpts1_f, kpts2_f, scores1_f, scores2_f, desc1_f, desc2_f, good, inlier_mask,
            thresholds
        )
        return {"match": "TRUE" if is_match else "FALSE", "diagnostics": diag}

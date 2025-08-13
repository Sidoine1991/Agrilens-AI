import os
import io
import time
import gc
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import HfFolder


APP_START_TS = time.time()

# Model configuration
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-3n-e2b-it")
DEVICE_MAP = os.environ.get("DEVICE_MAP", "auto")  # "auto" | "cpu" | "cuda"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))


def _get_dtype() -> torch.dtype:
    # Prefer float16 on CUDA, else bfloat16 if available, else float32
    if torch.cuda.is_available():
        return torch.float16
    # Many CPUs support bfloat16 with recent PyTorch; safe fallback
    try:
        _ = torch.tensor([1.0], dtype=torch.bfloat16)
        return torch.bfloat16
    except Exception:
        return torch.float32


def _build_prompt(culture: Optional[str], notes: Optional[str]) -> str:
    base = (
        "You are an agronomy assistant. Analyze the provided plant leaf image and identify the most likely disease. "
        "Return a concise diagnosis in French with: disease name, short explanation of symptoms, and 3 actionable treatment recommendations."
    )
    if culture:
        base += f"\nCulture: {culture}"
    if notes:
        base += f"\nNotes: {notes}"
    return base


class ModelBundle:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device_map = DEVICE_MAP
        self.dtype = _get_dtype()

    def load(self):
        if self.model is not None and self.processor is not None:
            return

        token = os.environ.get("HF_TOKEN") or HfFolder.get_token()
        common_args = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": self.device_map,
            "torch_dtype": self.dtype,
        }
        if token:
            common_args["token"] = token

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.processor = AutoProcessor.from_pretrained(MODEL_ID, **common_args)
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **common_args)


bundle = ModelBundle()
app = FastAPI(title="AgriLens AI FastAPI", version="1.0.0")


@app.on_event("startup")
def _startup():
    # Lazy load to reduce cold start impact; still attempt on startup
    try:
        bundle.load()
    except Exception:
        # Defer to first request error; health will report unloaded
        pass


@app.get("/health")
def health():
    return {
        "status": "ok" if (bundle.model is not None) else "cold",
        "uptime_s": int(time.time() - APP_START_TS),
        "cuda": torch.cuda.is_available(),
        "device_map": bundle.device_map,
        "dtype": str(bundle.dtype),
        "model_id": MODEL_ID,
    }


@app.post("/diagnose")
async def diagnose(
    image: UploadFile = File(..., description="Plant leaf image"),
    culture: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
):
    try:
        bundle.load()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

    try:
        raw = await image.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    prompt = _build_prompt(culture, notes)

    try:
        inputs = bundle.processor(text=prompt, images=img, return_tensors="pt")
        # Move to device if needed (device_map="auto" handles placements)
        if DEVICE_MAP == "cpu" or not torch.cuda.is_available():
            # Ensure tensors are on CPU for CPU-only runs
            inputs = {k: v.to("cpu") if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = bundle.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

        text = bundle.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        return JSONResponse(
            content={
                "success": True,
                "diagnosis": text.strip(),
                "meta": {
                    "model_id": MODEL_ID,
                    "dtype": str(bundle.dtype),
                },
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@app.post("/load")
def load():
    try:
        bundle.load()
        return {"status": "ok", "model_id": MODEL_ID}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load error: {e}")


@app.get("/")
def root():
    return {"message": "AgriLens AI FastAPI up. Use POST /diagnose"}



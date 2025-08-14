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

# Configuration du modèle
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-3n-e2b-it")
DEVICE_MAP = os.environ.get("DEVICE_MAP", "auto")  # "auto" | "cpu" | "cuda"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))

def _get_dtype() -> torch.dtype:
    """Choix optimal du dtype selon le hardware."""
    if torch.cuda.is_available():
        return torch.float16
    try:
        _ = torch.tensor([1.0], dtype=torch.bfloat16)
        return torch.bfloat16
    except Exception:
        return torch.float32

def _build_prompt(culture: Optional[str], notes: Optional[str]) -> str:
    """Création du prompt d'analyse."""
    base = (
        "You are an agronomy assistant. Analyze the provided plant leaf image and identify the most likely disease. "
        "Return a concise diagnosis in French with: disease name, short explanation of symptoms, "
        "and 3 actionable treatment recommendations."
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
        """Chargement du modèle et du processor."""
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

@app.get("/health")
def health():
    """Vérifie l'état de l'application et du modèle."""
    return {
        "status": "ok" if (bundle.model is not None) else "cold",
        "uptime_s": int(time.time() - APP_START_TS),
        "cuda": torch.cuda.is_available(),
        "device_map": bundle.device_map,
        "dtype": str(bundle.dtype),
        "model_id": MODEL_ID,
    }

@app.post("/load")
def load():
    """Charge le modèle manuellement."""
    try:
        bundle.load()
        return {"status": "ok", "model_id": MODEL_ID}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load error: {e}")

@app.post("/diagnose")
async def diagnose(
    image: UploadFile = File(..., description="Plant leaf image"),
    culture: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
):
    """Analyse une image de feuille et retourne un diagnostic."""
    try:
        bundle.load()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

    try:
        raw = await image.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        
        # Redimensionner l'image si nécessaire (comme dans Streamlit)
        if img.size[0] > 224 or img.size[1] > 224:
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Construire le prompt comme dans Streamlit
    additional_info = ""
    if culture and culture.strip():
        additional_info += f"Culture spécifiée : {culture.strip()}. "
    if notes and notes.strip():
        additional_info += f"Notes : {notes.strip()}. "

    # Construire les messages comme dans Streamlit
    system_message = f"""Tu es un expert en phytopathologie et agronomie. Analyse l'image fournie et fournis un diagnostic STRUCTURÉ et SUCCINCT en 3 parties obligatoires :

1. SYMPTÔMES VISIBLES : Description concise des signes visibles sur la plante (taches, déformations, jaunissement, etc.)

2. NOM DE LA MALADIE : Nom de la maladie la plus probable avec niveau de confiance (ex: "Mildiou du manioc (85%)")

3. TRAITEMENT RECOMMANDÉ : Actions concrètes à mettre en œuvre pour traiter la maladie (fongicide, amélioration ventilation, etc.)

{additional_info}

IMPORTANT : 
- Analyse UNIQUEMENT l'image fournie
- Sois précis et concis
- Inclus TOUJOURS les 3 sections
- Ne donne PAS de réponses génériques"""

    user_instruction = f"{additional_info}Analyse cette image de plante malade et fournis un diagnostic SUCCINCT et STRUCTURÉ."

    # Structure des messages pour Gemma 3n e2b it multimodal (comme dans Streamlit)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},  # Passer directement l'objet PIL Image
                {"type": "text", "text": f"{system_message}\n\n{user_instruction} IMPORTANT : Analyse uniquement ce que tu vois dans cette image spécifique. Ne donne pas de réponse générique."}
            ]
        }
    ]

    try:
        # Utiliser processor.apply_chat_template comme dans Streamlit
        try:
            inputs = bundle.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as template_error:
            # Fallback : essayer un format plus simple
            simple_prompt = f"{system_message}\n\n{user_instruction}"
            inputs = bundle.processor(simple_prompt, img, return_tensors="pt", padding=True, truncation=True)

        if DEVICE_MAP == "cpu" or not torch.cuda.is_available():
            inputs = {k: v.to("cpu") if hasattr(v, "to") else v for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = bundle.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=100,
                repetition_penalty=1.1,
                use_cache=True,
                num_beams=1
            )

        text = bundle.processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0]

        # Nettoyage des tokens de contrôle si présents
        final_response = text.strip()
        final_response = final_response.replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()

        return JSONResponse(
            content={
                "success": True,
                "diagnosis": final_response,
                "meta": {
                    "model_id": MODEL_ID,
                    "dtype": str(bundle.dtype),
                },
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

@app.get("/")
def root():
    return {"message": "AgriLens AI FastAPI up. Use POST /diagnose"}

# Lancement correct pour Hugging Face Spaces
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))  # Hugging Face donne ce port
    uvicorn.run("app:app", host="0.0.0.0", port=port)

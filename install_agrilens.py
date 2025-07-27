from huggingface_hub import hf_hub_download
import os

MODEL_DIR = "models/model_gemma"
os.makedirs(MODEL_DIR, exist_ok=True)

# Liste des fichiers à adapter selon le contenu réel de ton modèle
dataset_repo = "Sidoineko/data_gemma"
model_files = [
    "chat_template.jinja",
    "config",
    "generation_config",
    "model.safetensors.index",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "preprocessor_config",
    "processor_config",
    "README.md",
    "special_tokens_map",
    "tokenizer",
    "tokenizer.model",
    "tokenizer_config",
]

for fname in model_files:
    print(f"Téléchargement de {fname} depuis Hugging Face Datasets...")
    hf_hub_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        filename=f"model_gemma/{fname}",
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False
    )
print(f"Modèle prêt dans : {MODEL_DIR}")
# Si besoin, ajoute une vérification de présence de tous les fichiers 
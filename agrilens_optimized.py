# =================================================================================
# AgriLens AI - Version Optimisée pour Kaggle T4 GPU
# Auteur : Sidoine Kolaolé YEBADOKPO
# =================================================================================

import os
import torch
import time
import gc
from typing import Optional

# Configuration de l'environnement
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def memory_monitor(stage: str = ""):
    """Surveille l'utilisation de la mémoire GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        print(f"💾 Mémoire ({stage}): {allocated:.2f} GB / {reserved:.2f} GB")

def clear_memory():
    """Nettoie la mémoire"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def setup_and_install():
    """Configuration et installation des dépendances"""
    print("🔧 Configuration de l'environnement...")
    
    # Installation des dépendances
    import subprocess
    import sys
    
    packages = [
        "timm --upgrade -q",
        "accelerate -q", 
        "git+https://github.com/huggingface/transformers.git --upgrade -q"
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + package.split())
    
    print("✅ Dépendances installées")
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA: {torch.cuda.is_available()}")

class AgriLensModel:
    """Gestionnaire de modèle AgriLens AI"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.processor = None
        self.multimodal_model = None
    
    def load_text_model(self) -> bool:
        """Charge le modèle de texte"""
        try:
            print("📝 Chargement du modèle de texte...")
            
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            memory_monitor("Modèle texte chargé")
            print("✅ Modèle de texte prêt")
            return True
            
        except Exception as e:
            print(f"❌ Erreur modèle texte: {e}")
            return False
    
    def load_multimodal_model(self) -> bool:
        """Charge le modèle multimodal"""
        try:
            print("🖼️ Chargement du modèle multimodal...")
            
            from transformers import AutoProcessor, AutoModelForImageTextToText
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            
            self.multimodal_model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="cpu",
                torch_dtype=torch.float32
            )
            
            print("✅ Modèle multimodal prêt")
            return True
            
        except Exception as e:
            print(f"❌ Erreur modèle multimodal: {e}")
            return False

def generate_text(model: AgriLensModel, prompt: str, max_tokens: int = 200) -> Optional[str]:
    """Génère du texte"""
    try:
        print(f"🔤 Génération: {prompt[:50]}...")
        
        inputs = model.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        
        device = next(model.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=model.tokenizer.eos_token_id
            )
        
        response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        print(f"⏱️ {time.time() - start_time:.2f}s - {len(response)} caractères")
        return response
        
    except Exception as e:
        print(f"❌ Erreur génération: {e}")
        return None

def analyze_image(model: AgriLensModel, image_path: str) -> Optional[str]:
    """Analyse une image"""
    try:
        print(f"🖼️ Analyse: {image_path}")
        
        from PIL import Image
        
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        
        prompt = (
            "Analyse cette image de plante. Décris les symptômes et fournis un diagnostic "
            "structuré: 1) Maladie probable, 2) Symptômes, 3) Causes, 4) Traitements, 5) Prévention"
        )
        
        inputs = model.processor(text=prompt, images=image, return_tensors="pt")
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.multimodal_model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7
            )
        
        response = model.processor.decode(outputs[0], skip_special_tokens=True)
        
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        print(f"⏱️ {time.time() - start_time:.2f}s - {len(response)} caractères")
        return response
        
    except Exception as e:
        print(f"❌ Erreur analyse: {e}")
        return None

def main():
    """Fonction principale"""
    print("🚀 AgriLens AI - Version Optimisée")
    print("=" * 50)
    
    # Setup
    setup_and_install()
    
    # Chemin du modèle
    try:
        import kagglehub
        GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")
    except:
        GEMMA_PATH = "/kaggle/input/gemma-3n/transformers/gemma-3n-e2b-it/1"
    
    print(f"📁 Modèle: {GEMMA_PATH}")
    
    # Initialisation
    agrilens = AgriLensModel(GEMMA_PATH)
    
    # Chargement des modèles
    text_ok = agrilens.load_text_model()
    multimodal_ok = agrilens.load_multimodal_model()
    
    if not text_ok and not multimodal_ok:
        print("❌ Aucun modèle chargé")
        return
    
    # Tests
    print("\n🧪 TESTS")
    print("=" * 30)
    
    # Test texte
    if text_ok:
        print("\n📝 Test génération texte:")
        text_prompt = """
        Décris les symptômes du mildiou chez les tomates, 
        incluant l'apparence et la progression de la maladie.
        """
        
        response = generate_text(agrilens, text_prompt)
        if response:
            print("📋 Réponse:")
            print("-" * 30)
            print(response)
            print("-" * 30)
    
    # Test image
    if multimodal_ok:
        print("\n🖼️ Test analyse image:")
        
        test_images = [
            "/kaggle/input/tomato/tomato_early_blight.jpg",
            "/kaggle/input/tomato/tomato_late_blight.jpg",
            "/kaggle/input/tomato/tomato_healthy.jpg"
        ]
        
        for img_path in test_images:
            if os.path.exists(img_path):
                response = analyze_image(agrilens, img_path)
                if response:
                    print("📋 Diagnostic:")
                    print("-" * 30)
                    print(response)
                    print("-" * 30)
                break
    
    # Nettoyage
    clear_memory()
    print("\n🎉 Terminé!")

if __name__ == "__main__":
    main() 
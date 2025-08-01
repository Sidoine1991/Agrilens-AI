# =================================================================================
# AgriLens AI - Version Améliorée pour Kaggle T4 GPU
# Auteur : Sidoine Kolaolé YEBADOKPO
# Optimisé pour la génération de texte et l'analyse d'images
# =================================================================================

import os
import torch
import time
import gc
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# =================================================================================
# CONFIGURATION ET SETUP
# =================================================================================

def setup_environment():
    """Configure l'environnement pour l'exécution optimale"""
    print("🔧 Configuration de l'environnement...")
    
    # Forcer l'utilisation d'un seul GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print(f"✅ CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

def install_dependencies():
    """Installe les dépendances nécessaires"""
    print("\n📦 Installation des dépendances...")
    
    try:
        import subprocess
        import sys
        
        packages = [
            "timm",
            "accelerate", 
            "git+https://github.com/huggingface/transformers.git"
        ]
        
        for package in packages:
            print(f"📥 Installation de {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--upgrade", "-q"
            ])
        
        print("✅ Toutes les dépendances installées")
        
    except Exception as e:
        print(f"⚠️ Erreur lors de l'installation: {e}")

def memory_monitor(stage: str = ""):
    """Surveille l'utilisation de la mémoire GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        print(f"💾 Mémoire ({stage}): Allouée: {allocated:.2f} GB, Réservée: {reserved:.2f} GB")

def clear_memory():
    """Nettoie la mémoire GPU et CPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("🧹 Mémoire nettoyée")

# =================================================================================
# CHARGEMENT DU MODÈLE
# =================================================================================

class AgriLensModel:
    """Classe pour gérer le modèle AgriLens AI"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.processor = None
        self.multimodal_model = None
        
    def load_text_model(self) -> bool:
        """Charge le modèle de génération de texte"""
        try:
            print("\n📝 Chargement du modèle de texte...")
            
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Chargement du tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            # Configuration du tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Chargement du modèle avec optimisations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            memory_monitor("Après chargement texte")
            print("✅ Modèle de texte chargé avec succès")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle de texte: {e}")
            return False
    
    def load_multimodal_model(self) -> bool:
        """Charge le modèle multimodal pour l'analyse d'images"""
        try:
            print("\n🖼️ Chargement du modèle multimodal...")
            
            from transformers import AutoProcessor, AutoModelForImageTextToText
            
            # Chargement du processeur
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Chargement du modèle multimodal (CPU pour économiser la mémoire)
            self.multimodal_model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="cpu",
                torch_dtype=torch.float32
            )
            
            print("✅ Modèle multimodal chargé avec succès")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle multimodal: {e}")
            return False

# =================================================================================
# GÉNÉRATION DE TEXTE
# =================================================================================

def generate_text_response(
    model: AgriLensModel,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7
) -> Optional[str]:
    """Génère une réponse textuelle"""
    
    try:
        print(f"\n🔤 Génération de texte...")
        print(f"📝 Prompt: {prompt[:100]}...")
        
        if model.tokenizer is None or model.model is None:
            print("❌ Modèle de texte non chargé")
            return None
        
        # Tokenisation
        inputs = model.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Déplacer vers le bon device
        device = next(model.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        memory_monitor("Avant génération")
        
        # Génération
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=model.tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Décodage
        response = model.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        # Nettoyage de la réponse
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        memory_monitor("Après génération")
        
        print(f"⏱️ Temps de génération: {generation_time:.2f}s")
        print(f"📝 Réponse générée ({len(response)} caractères)")
        
        return response
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération de texte: {e}")
        return None

# =================================================================================
# ANALYSE D'IMAGES
# =================================================================================

def analyze_plant_image(
    model: AgriLensModel,
    image_path: str,
    prompt: str = None
) -> Optional[str]:
    """Analyse une image de plante"""
    
    try:
        print(f"\n🖼️ Analyse d'image: {image_path}")
        
        if model.processor is None or model.multimodal_model is None:
            print("❌ Modèle multimodal non chargé")
            return None
        
        # Chargement et préparation de l'image
        from PIL import Image
        
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))  # Redimensionnement pour économiser la mémoire
        
        print(f"✅ Image chargée: {image.size}")
        
        # Prompt par défaut si aucun fourni
        if prompt is None:
            prompt = (
                "Analyse cette image de plante. Décris les symptômes visibles et "
                "fournis un diagnostic structuré incluant:\n"
                "1. Nom de la maladie probable\n"
                "2. Symptômes observés\n"
                "3. Causes possibles\n"
                "4. Traitements recommandés\n"
                "5. Mesures préventives"
            )
        
        # Préparation des inputs
        inputs = model.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        
        # Génération
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.multimodal_model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        generation_time = time.time() - start_time
        
        # Décodage
        response = model.processor.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        # Nettoyage
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        print(f"⏱️ Temps d'analyse: {generation_time:.2f}s")
        print(f"📋 Diagnostic généré ({len(response)} caractères)")
        
        return response
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse d'image: {e}")
        return None

# =================================================================================
# FONCTION PRINCIPALE
# =================================================================================

def main():
    """Fonction principale d'exécution"""
    
    print("🚀 AgriLens AI - Version Améliorée")
    print("=" * 60)
    
    # 1. Configuration de l'environnement
    setup_environment()
    
    # 2. Installation des dépendances
    install_dependencies()
    
    # 3. Import des bibliothèques
    print("\n📚 Import des bibliothèques...")
    try:
        import kagglehub
        from transformers import GenerationConfig
        print("✅ Bibliothèques importées")
    except Exception as e:
        print(f"❌ Erreur d'import: {e}")
        return
    
    # 4. Chemin du modèle
    try:
        GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")
        print(f"✅ Modèle localisé: {GEMMA_PATH}")
    except:
        # Fallback vers le chemin local
        GEMMA_PATH = "/kaggle/input/gemma-3n/transformers/gemma-3n-e2b-it/1"
        print(f"✅ Utilisation du chemin local: {GEMMA_PATH}")
    
    # 5. Initialisation du modèle
    agrilens = AgriLensModel(GEMMA_PATH)
    
    # 6. Chargement des modèles
    text_success = agrilens.load_text_model()
    multimodal_success = agrilens.load_multimodal_model()
    
    if not text_success and not multimodal_success:
        print("❌ Aucun modèle n'a pu être chargé")
        return
    
    # 7. Tests de génération
    print("\n" + "="*60)
    print("🧪 TESTS DE GÉNÉRATION")
    print("="*60)
    
    # Test de génération de texte
    if text_success:
        print("\n📝 Test de génération de texte...")
        
        text_prompt = """
        Fournis une description détaillée des symptômes de la maladie du mildiou 
        chez les plants de tomate. Inclus des informations sur l'apparence, 
        la progression et les parties spécifiques de la plante affectées.
        """
        
        text_response = generate_text_response(
            agrilens, 
            text_prompt, 
            max_new_tokens=200
        )
        
        if text_response:
            print("\n📋 Réponse du modèle:")
            print("-" * 40)
            print(text_response)
            print("-" * 40)
    
    # Test d'analyse d'image
    if multimodal_success:
        print("\n🖼️ Test d'analyse d'image...")
        
        # Chercher une image de test
        test_images = [
            "/kaggle/input/tomato/tomato_early_blight.jpg",
            "/kaggle/input/tomato/tomato_late_blight.jpg",
            "/kaggle/input/tomato/tomato_healthy.jpg"
        ]
        
        image_found = False
        for image_path in test_images:
            if os.path.exists(image_path):
                image_response = analyze_plant_image(agrilens, image_path)
                if image_response:
                    print("\n📋 Diagnostic d'image:")
                    print("-" * 40)
                    print(image_response)
                    print("-" * 40)
                    image_found = True
                    break
        
        if not image_found:
            print("⚠️ Aucune image de test trouvée")
    
    # 8. Nettoyage final
    clear_memory()
    
    print("\n" + "="*60)
    print("🎉 AgriLens AI - Exécution terminée avec succès!")
    print("="*60)

if __name__ == "__main__":
    main() 
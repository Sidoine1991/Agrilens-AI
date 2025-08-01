# =================================================================================
# Test de Génération - Gemma 3n E2B IT (Version avec Transformers à jour)
# 
# Auteur : Sidoine Kolaolé YEBADOKPO
# Objectif : Tester les capacités de génération avec Transformers mis à jour
# =================================================================================

import os
import torch
import time
import subprocess
import sys

# Installation des dépendances avec Transformers à jour
print("🔧 Installation des dépendances avec Transformers à jour...")

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installé avec succès")
    except Exception as e:
        print(f"⚠️ Erreur installation {package}: {e}")

# Installation de Transformers depuis la source (nécessaire pour Gemma 3n)
print("📦 Installation de Transformers depuis la source...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/huggingface/transformers.git"])
    print("✅ Transformers installé depuis la source")
except Exception as e:
    print(f"⚠️ Erreur installation Transformers: {e}")

# Installation des autres dépendances
packages = [
    "bitsandbytes",
    "accelerate",
    "torch",
    "torchvision",
    "pillow",
    "requests"
]

for package in packages:
    install_package(package)

print("\n📚 Import des bibliothèques...")

# Import après installation
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from PIL import Image
import requests
from io import BytesIO

# Configuration de l'environnement
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Chemin local du modèle fourni par Kaggle
MODEL_PATH = "/kaggle/input/gemma-3n/transformers/gemma-3n-e2b-it/1"

def load_model():
    """Charge le modèle Gemma 3n avec la configuration optimisée"""
    try:
        print("🔄 Chargement du modèle Gemma 3n E2B IT...")
        print(f"📁 Chemin du modèle: {MODEL_PATH}")
        
        # Vérifier si le chemin existe
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Le chemin {MODEL_PATH} n'existe pas!")
            return None, None
        
        print("✅ Chemin du modèle vérifié")
        
        # Configuration 4-bit pour économiser la mémoire
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_block_size=16,
        )
        
        # Chargement du processeur
        print("📝 Chargement du processeur...")
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
        # Chargement du modèle
        print("🤖 Chargement du modèle...")
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_config,
        )
        
        print("✅ Modèle chargé avec succès!")
        return model, processor
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        print(f"Type d'erreur: {type(e).__name__}")
        return None, None

def test_text_generation(model, processor, prompt, max_new_tokens=100):
    """Teste la génération de texte avec un prompt simple"""
    try:
        print(f"\n🔤 Test de génération de texte:")
        print(f"Prompt: {prompt}")
        print("-" * 50)
        
        # Préparation des inputs
        inputs = processor(text=prompt, return_tensors="pt")
        
        # Mesure du temps de génération
        start_time = time.time()
        
        # Génération
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Décodage de la réponse
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Nettoyage de la réponse (enlever le prompt original)
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        print(f"⏱️ Temps de génération: {generation_time:.2f} secondes")
        print(f"📝 Réponse générée:")
        print(response)
        print("-" * 50)
        
        return response, generation_time
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {e}")
        return None, 0

def test_multimodal_generation(model, processor, image, prompt, max_new_tokens=100):
    """Teste la génération multimodale (image + texte)"""
    try:
        print(f"\n🖼️ Test de génération multimodale:")
        print(f"Prompt: {prompt}")
        print(f"Image: {image.size}")
        print("-" * 50)
        
        # Préparation des inputs avec image
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        # Mesure du temps de génération
        start_time = time.time()
        
        # Génération
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Décodage de la réponse
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Nettoyage de la réponse
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        print(f"⏱️ Temps de génération: {generation_time:.2f} secondes")
        print(f"📝 Réponse générée:")
        print(response)
        print("-" * 50)
        
        return response, generation_time
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération multimodale: {e}")
        return None, 0

def load_test_image():
    """Charge une image de test pour la démonstration"""
    try:
        # URL d'une image de plante pour test
        image_url = "https://images.unsplash.com/photo-1546094096-0df4bcaaa337?w=400"
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        print(f"✅ Image de test chargée: {image.size}")
        return image
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'image: {e}")
        return None

def main():
    """Fonction principale de test"""
    print("🚀 Démarrage des tests de génération Gemma 3n")
    print("=" * 60)
    
    # Vérification de l'environnement
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"💻 CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Chargement du modèle
    model, processor = load_model()
    
    if model is None or processor is None:
        print("❌ Impossible de continuer sans modèle chargé")
        return
    
    # Chargement d'une image de test
    test_image = load_test_image()
    
    # Test 1: Génération de texte simple
    print("\n" + "="*60)
    print("TEST 1: GÉNÉRATION DE TEXTE SIMPLE")
    print("="*60)
    
    simple_prompts = [
        "Qu'est-ce que l'intelligence artificielle?",
        "Explique-moi l'agriculture durable en 3 points.",
        "Comment les plantes communiquent-elles entre elles?"
    ]
    
    for prompt in simple_prompts:
        test_text_generation(model, processor, prompt)
    
    # Test 2: Génération multimodale (si image disponible)
    if test_image:
        print("\n" + "="*60)
        print("TEST 2: GÉNÉRATION MULTIMODALE")
        print("="*60)
        
        multimodal_prompts = [
            "Décris ce que tu vois dans cette image.",
            "Cette plante semble-t-elle en bonne santé?",
            "Quels conseils donnerais-tu pour cette plante?"
        ]
        
        for prompt in multimodal_prompts:
            test_multimodal_generation(model, processor, test_image, prompt)
    
    # Test 3: Diagnostic de plantes
    print("\n" + "="*60)
    print("TEST 3: DIAGNOSTIC DE PLANTES")
    print("="*60)
    
    if test_image:
        plant_prompts = [
            "Analyse cette image de plante et identifie les problèmes visibles.",
            "Décris les symptômes visibles sur cette plante et leurs causes possibles.",
            "Basé sur cette image, quels traitements recommandes-tu pour cette plante?"
        ]
        
        for prompt in plant_prompts:
            test_multimodal_generation(model, processor, test_image, prompt)
    else:
        # Test avec texte seulement
        text_prompts = [
            "Comment identifier les maladies des plantes?",
            "Quels sont les symptômes courants des carences en nutriments?",
            "Comment prévenir les maladies fongiques chez les plantes?"
        ]
        
        for prompt in text_prompts:
            test_text_generation(model, processor, prompt)
    
    print("\n" + "="*60)
    print("🎉 Tests terminés avec succès!")
    print("="*60)

if __name__ == "__main__":
    main() 
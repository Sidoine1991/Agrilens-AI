# =================================================================================
# Test Simple - Gemma 3n E2B IT (Transformers à jour)
# Version simple avec Transformers mis à jour depuis la source
# =================================================================================

import os
import torch
import time

# Installation de Transformers depuis la source (CRUCIAL pour Gemma 3n)
print("🔧 Installation de Transformers depuis la source...")
os.system("pip install git+https://github.com/huggingface/transformers.git")

# Installation des autres dépendances
print("📦 Installation des autres dépendances...")
os.system("pip install bitsandbytes accelerate torch pillow requests")

# Import après installation
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_PATH = "/kaggle/input/gemma-3n/transformers/gemma-3n-e2b-it/1"

def simple_test():
    """Test simple de génération avec Transformers à jour"""
    
    print("🚀 Test simple Gemma 3n (Transformers à jour)")
    print("=" * 50)
    
    try:
        # 1. Vérification du chemin
        print(f"📁 Vérification du chemin: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Chemin inexistant: {MODEL_PATH}")
            return False
        
        # 2. Chargement du modèle
        print("📥 Chargement du modèle...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_config,
        )
        
        print("✅ Modèle chargé!")
        
        # 3. Test de génération
        print("\n🔤 Test de génération...")
        
        prompt = "Explique-moi l'agriculture durable en 2 phrases."
        print(f"Prompt: {prompt}")
        
        inputs = processor(text=prompt, return_tensors="pt")
        
        start_time = time.time()
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
            )
        
        gen_time = time.time() - start_time
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        print(f"\n⏱️ Temps: {gen_time:.2f}s")
        print(f"📝 Réponse: {response}")
        
        print("\n🎉 Test réussi!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print(f"Type d'erreur: {type(e).__name__}")
        return False

if __name__ == "__main__":
    simple_test() 
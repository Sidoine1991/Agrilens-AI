# =================================================================================
# Test Minimal de Génération - Gemma 3n E2B IT
# Version ultra-simple pour test rapide
# =================================================================================

import os
import torch
import time

# Installation rapide des dépendances
print("🔧 Installation des dépendances...")
os.system("pip install bitsandbytes accelerate transformers torch")

# Import après installation
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_PATH = "/kaggle/input/gemma-3n/transformers/gemma-3n-e2b-it/1"

def minimal_test():
    """Test minimal de génération"""
    
    print("🚀 Test minimal Gemma 3n")
    print("=" * 40)
    
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
        
        prompt = "Explique-moi l'agriculture durable."
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
        return False

if __name__ == "__main__":
    minimal_test() 
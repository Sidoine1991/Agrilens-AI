# =================================================================================
# Test Rapide de Génération de Texte - Gemma 3n E2B IT
# Version simplifiée pour test rapide
# =================================================================================

import os
import torch
import time
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_PATH = "/kaggle/input/gemma-3n/transformers/gemma-3n-e2b-it/1"

def quick_test():
    """Test rapide de génération de texte"""
    
    print("🚀 Test rapide de génération Gemma 3n")
    print("=" * 50)
    
    try:
        # 1. Chargement du modèle
        print("📥 Chargement du modèle...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_block_size=16,
        )
        
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_config,
        )
        
        print("✅ Modèle chargé!")
        
        # 2. Test de génération simple
        print("\n🔤 Test de génération de texte...")
        
        prompt = "Explique-moi l'agriculture durable en 3 points clés."
        print(f"Prompt: {prompt}")
        
        # Préparation
        inputs = processor(text=prompt, return_tensors="pt")
        
        # Génération
        start_time = time.time()
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        gen_time = time.time() - start_time
        
        # Décodage
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Nettoyage
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        print(f"\n⏱️ Temps: {gen_time:.2f}s")
        print(f"📝 Réponse:")
        print(response)
        
        print("\n🎉 Test réussi!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    quick_test() 
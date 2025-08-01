# =================================================================================
# Test de Diagnostic de Plantes - Gemma 3n E2B IT
# Spécialisé pour l'application AgriLens AI
# =================================================================================

import os
import torch
import time
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from PIL import Image
import requests
from io import BytesIO

# Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_PATH = "/kaggle/input/gemma-3n/transformers/gemma-3n-e2b-it/1"

def load_model():
    """Charge le modèle pour le diagnostic"""
    try:
        print("🌱 Chargement du modèle pour diagnostic de plantes...")
        
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
        
        print("✅ Modèle chargé pour diagnostic!")
        return model, processor
        
    except Exception as e:
        print(f"❌ Erreur de chargement: {e}")
        return None, None

def load_plant_image():
    """Charge une image de plante pour test"""
    try:
        # Image de tomate avec maladie (exemple)
        image_url = "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400"
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        print(f"✅ Image de plante chargée: {image.size}")
        return image
    except Exception as e:
        print(f"❌ Erreur image: {e}")
        return None

def diagnose_plant(model, processor, image, prompt):
    """Effectue un diagnostic de plante"""
    try:
        print(f"\n🔍 Diagnostic: {prompt}")
        print("-" * 40)
        
        # Préparation avec image
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        # Génération
        start_time = time.time()
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
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
        
        print(f"⏱️ Temps: {gen_time:.2f}s")
        print(f"📋 Diagnostic:")
        print(response)
        print("-" * 40)
        
        return response, gen_time
        
    except Exception as e:
        print(f"❌ Erreur diagnostic: {e}")
        return None, 0

def test_plant_diagnosis():
    """Test complet du diagnostic de plantes"""
    
    print("🌱 Test de Diagnostic de Plantes - AgriLens AI")
    print("=" * 60)
    
    # 1. Chargement
    model, processor = load_model()
    if not model or not processor:
        print("❌ Impossible de continuer")
        return
    
    # 2. Chargement image
    plant_image = load_plant_image()
    if not plant_image:
        print("❌ Impossible de charger l'image")
        return
    
    # 3. Tests de diagnostic
    diagnostic_tests = [
        {
            "type": "Analyse générale",
            "prompt": "Analyse cette image de plante et identifie les problèmes visibles."
        },
        {
            "type": "Symptômes",
            "prompt": "Décris les symptômes visibles sur cette plante et leurs causes possibles."
        },
        {
            "type": "Traitement",
            "prompt": "Basé sur cette image, quels traitements recommandes-tu pour cette plante?"
        },
        {
            "type": "Prévention",
            "prompt": "Comment peut-on prévenir ces problèmes à l'avenir?"
        }
    ]
    
    results = []
    
    for test in diagnostic_tests:
        print(f"\n🔬 {test['type']}")
        response, gen_time = diagnose_plant(model, processor, plant_image, test['prompt'])
        
        if response:
            results.append({
                'type': test['type'],
                'response': response,
                'time': gen_time,
                'success': True
            })
        else:
            results.append({
                'type': test['type'],
                'response': 'Échec',
                'time': 0,
                'success': False
            })
    
    # 4. Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES DIAGNOSTICS")
    print("=" * 60)
    
    successful = sum(1 for r in results if r['success'])
    total_time = sum(r['time'] for r in results if r['success'])
    
    print(f"✅ Diagnostics réussis: {successful}/{len(results)}")
    
    if successful > 0:
        avg_time = total_time / successful
        print(f"⏱️ Temps moyen: {avg_time:.2f}s")
        print(f"⏱️ Temps total: {total_time:.2f}s")
    
    print("\n🎉 Test de diagnostic terminé!")

if __name__ == "__main__":
    test_plant_diagnosis() 
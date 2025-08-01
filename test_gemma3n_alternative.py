# =================================================================================
# Test Alternative - Gemma 3n E2B IT
# Approches alternatives pour charger le modèle Gemma 3n
# =================================================================================

import os
import torch
import time
import json

# Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_PATH = "/kaggle/input/gemma-3n/transformers/gemma-3n-e2b-it/1"

def check_model_files():
    """Vérifie les fichiers disponibles dans le modèle"""
    print("🔍 Vérification des fichiers du modèle...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Le chemin {MODEL_PATH} n'existe pas!")
        return False
    
    print(f"✅ Chemin existe: {MODEL_PATH}")
    
    # Lister les fichiers
    try:
        files = os.listdir(MODEL_PATH)
        print(f"📁 Fichiers trouvés: {files}")
        
        # Vérifier le fichier config.json
        config_path = os.path.join(MODEL_PATH, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"📋 Configuration du modèle:")
            print(f"  • Model type: {config.get('model_type', 'Non spécifié')}")
            print(f"  • Architecture: {config.get('architectures', 'Non spécifié')}")
            print(f"  • Vocab size: {config.get('vocab_size', 'Non spécifié')}")
            print(f"  • Hidden size: {config.get('hidden_size', 'Non spécifié')}")
            print(f"  • Num layers: {config.get('num_hidden_layers', 'Non spécifié')}")
        else:
            print("❌ Fichier config.json non trouvé")
            
    except Exception as e:
        print(f"❌ Erreur lors de la vérification: {e}")
        return False
    
    return True

def try_different_loading_approaches():
    """Essaie différentes approches pour charger le modèle"""
    
    print("\n🔄 Tentative de chargement avec différentes approches...")
    
    # Approche 1: AutoProcessor + AutoModelForCausalLM
    print("\n📝 Approche 1: AutoProcessor + AutoModelForCausalLM")
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_config,
        )
        
        print("✅ Approche 1 réussie!")
        return model, processor
        
    except Exception as e:
        print(f"❌ Approche 1 échouée: {e}")
    
    # Approche 2: AutoTokenizer + AutoModelForCausalLM
    print("\n📝 Approche 2: AutoTokenizer + AutoModelForCausalLM")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_config,
        )
        
        print("✅ Approche 2 réussie!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Approche 2 échouée: {e}")
    
    # Approche 3: Chargement sans quantification
    print("\n📝 Approche 3: Chargement sans quantification")
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        print("✅ Approche 3 réussie!")
        return model, processor
        
    except Exception as e:
        print(f"❌ Approche 3 échouée: {e}")
    
    # Approche 4: Chargement CPU seulement
    print("\n📝 Approche 4: Chargement CPU seulement")
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        
        print("✅ Approche 4 réussie!")
        return model, processor
        
    except Exception as e:
        print(f"❌ Approche 4 échouée: {e}")
    
    return None, None

def test_generation_with_processor(model, processor, prompt="Explique-moi l'agriculture durable."):
    """Test de génération avec un processeur"""
    try:
        print(f"\n🔤 Test de génération avec processeur:")
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
        
        print(f"⏱️ Temps: {gen_time:.2f}s")
        print(f"📝 Réponse: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur génération: {e}")
        return False

def test_generation_with_tokenizer(model, tokenizer, prompt="Explique-moi l'agriculture durable."):
    """Test de génération avec un tokenizer"""
    try:
        print(f"\n🔤 Test de génération avec tokenizer:")
        print(f"Prompt: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        start_time = time.time()
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
            )
        
        gen_time = time.time() - start_time
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        print(f"⏱️ Temps: {gen_time:.2f}s")
        print(f"📝 Réponse: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur génération: {e}")
        return False

def main():
    """Fonction principale"""
    print("🚀 Test Alternative - Gemma 3n E2B IT")
    print("=" * 60)
    
    # Vérification des fichiers
    if not check_model_files():
        print("❌ Impossible de continuer")
        return
    
    # Tentative de chargement
    model, processor_or_tokenizer = try_different_loading_approaches()
    
    if model is None:
        print("\n❌ Aucune approche n'a fonctionné")
        print("💡 Suggestions:")
        print("  • Le modèle Gemma 3n est très récent et n'est pas encore supporté")
        print("  • Attendez une mise à jour de Transformers")
        print("  • Utilisez un modèle alternatif (Gemma 2B, Llama, etc.)")
        return
    
    # Test de génération
    print("\n🎯 Test de génération...")
    
    if hasattr(processor_or_tokenizer, 'decode'):
        # C'est un processeur
        success = test_generation_with_processor(model, processor_or_tokenizer)
    else:
        # C'est un tokenizer
        success = test_generation_with_tokenizer(model, processor_or_tokenizer)
    
    if success:
        print("\n🎉 Test réussi!")
    else:
        print("\n❌ Test échoué")

if __name__ == "__main__":
    main() 
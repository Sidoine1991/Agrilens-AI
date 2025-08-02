#!/usr/bin/env python3
"""
Script de test des performances pour AgriLens AI
Teste les différents modes de performance et mesure les temps de réponse.
"""

import time
import torch
import os
import sys
from PIL import Image
import numpy as np

def test_performance_modes():
    """Teste les différents modes de performance."""
    
    print("🚀 Test des performances AgriLens AI")
    print("=" * 50)
    
    # Vérifier que le modèle local existe
    LOCAL_MODEL_PATH = "D:/Dev/model_gemma"
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"❌ Modèle local non trouvé : {LOCAL_MODEL_PATH}")
        print("💡 Assurez-vous que le modèle est téléchargé et configuré")
        return False
    
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        print("📦 Chargement du modèle...")
        
        # Configuration optimisée
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
        
        print("✅ Modèle chargé avec succès")
        
        # Créer une image de test simple
        test_image = Image.new('RGB', (224, 224), color='green')
        
                        # Configurations de test
                test_configs = {
                    "fast": {"max_new_tokens": 250, "top_k": 50},
                    "balanced": {"max_new_tokens": 300, "top_k": 100},
                    "quality": {"max_new_tokens": 350, "top_k": 200}
                }
        
        # Test avec une image
        print("\n📸 Test d'analyse d'image :")
        print("-" * 30)
        
        for mode, config in test_configs.items():
            print(f"\n🔍 Test mode {mode.upper()} :")
            print(f"   • max_new_tokens: {config['max_new_tokens']}")
            print(f"   • top_k: {config['top_k']}")
            
            # Préparer les inputs
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "Tu es un expert en pathologie végétale."}]},
                {"role": "user", "content": [
                    {"type": "image", "image": test_image},
                    {"type": "text", "text": "Analyse cette image de plante."}
                ]}
            ]
            
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            device = getattr(model, 'device', 'cpu')
            if hasattr(inputs, 'to'):
                inputs = inputs.to(device)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Mesurer le temps
            start_time = time.time()
            
            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    max_new_tokens=config['max_new_tokens'],
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=config['top_k'],
                    repetition_penalty=1.1,
                    use_cache=True,
                    num_beams=1,
                )
                
                response = processor.decode(generation[0][input_len:], skip_special_tokens=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"   ⏱️  Temps de réponse : {duration:.2f} secondes")
            print(f"   📝 Tokens générés : {len(response.split())} mots")
            print(f"   🚀 Vitesse : {config['max_new_tokens']/duration:.1f} tokens/seconde")
        
        # Test avec du texte
        print("\n📝 Test d'analyse de texte :")
        print("-" * 30)
        
        test_text = "Les feuilles de ma plante de tomate ont des taches jaunes et brunes."
        
        for mode, config in test_configs.items():
            print(f"\n🔍 Test mode {mode.upper()} :")
            
            prompt_template = f"Tu es un assistant agricole expert. Analyse ce problème : {test_text}"
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt_template}]}]
            
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            device = getattr(model, 'device', 'cpu')
            if hasattr(inputs, 'to'):
                inputs = inputs.to(device)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Mesurer le temps
            start_time = time.time()
            
            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    max_new_tokens=config['max_new_tokens'],
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=config['top_k'],
                    repetition_penalty=1.1,
                    use_cache=True,
                    num_beams=1,
                )
                
                response = processor.decode(generation[0][input_len:], skip_special_tokens=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"   ⏱️  Temps de réponse : {duration:.2f} secondes")
            print(f"   📝 Tokens générés : {len(response.split())} mots")
            print(f"   🚀 Vitesse : {config['max_new_tokens']/duration:.1f} tokens/seconde")
        
        print("\n✅ Tests de performance terminés !")
        print("\n💡 Recommandations :")
        print("   • Mode FAST : Pour les diagnostics rapides")
        print("   • Mode BALANCED : Pour un bon équilibre vitesse/qualité")
        print("   • Mode QUALITY : Pour les analyses détaillées")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test : {e}")
        return False

def test_memory_usage():
    """Teste l'utilisation mémoire."""
    print("\n💾 Test d'utilisation mémoire :")
    print("-" * 30)
    
    try:
        import psutil
        
        # Avant chargement
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        print(f"📊 Mémoire avant chargement : {memory_before:.1f} MB")
        
        # Charger le modèle
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        LOCAL_MODEL_PATH = "D:/Dev/model_gemma"
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        # Après chargement
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        print(f"📊 Mémoire après chargement : {memory_after:.1f} MB")
        print(f"📊 Mémoire utilisée par le modèle : {memory_used:.1f} MB")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            print(f"📊 Mémoire GPU utilisée : {gpu_memory:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test mémoire : {e}")
        return False

def main():
    """Fonction principale."""
    print("🔬 Test complet des performances AgriLens AI")
    print("=" * 60)
    
    # Test des performances
    if test_performance_modes():
        print("\n✅ Tests de performance réussis")
    else:
        print("\n❌ Tests de performance échoués")
    
    # Test de mémoire
    if test_memory_usage():
        print("\n✅ Tests de mémoire réussis")
    else:
        print("\n❌ Tests de mémoire échoués")
    
    print("\n🎯 Optimisations appliquées :")
    print("   • Quantisation 4-bit avec double quantisation")
    print("   • Flash Attention 2 (si disponible)")
    print("   • Paramètres de génération optimisés")
    print("   • Cache activé pour la génération")
    print("   • Modes de performance configurables")

if __name__ == "__main__":
    main() 
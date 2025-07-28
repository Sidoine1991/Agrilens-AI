#!/usr/bin/env python3
"""
Script de test pour le chargement du modèle Gemma 3n E4B IT
Teste différentes stratégies de chargement pour éviter l'erreur de disk_offload
"""

import torch
import sys
import os

def test_memory_availability():
    """Teste la disponibilité de la mémoire"""
    print("🔍 Vérification de la mémoire disponible...")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU disponible : {gpu_memory:.1f} GB")
        return gpu_memory
    else:
        print("⚠️ GPU non disponible, utilisation du CPU")
        return 0

def test_model_loading():
    """Teste le chargement du modèle avec différentes stratégies"""
    print("\n🚀 Test de chargement du modèle Gemma 3n E4B IT...")
    
    try:
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration
        
        model_id = "google/gemma-3n-E4B-it"
        
        # Charger le processeur
        print("📥 Chargement du processeur...")
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        print("✅ Processeur chargé avec succès")
        
        # Stratégies de chargement
        strategies = [
            ("CPU Conservateur", lambda: Gemma3nForConditionalGeneration.from_pretrained(
                model_id,
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={"cpu": "8GB"}
            )),
            ("4-bit Quantization", lambda: Gemma3nForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )),
            ("8-bit Quantization", lambda: Gemma3nForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                load_in_8bit=True
            )),
            ("Gestion mémoire personnalisée", lambda: Gemma3nForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "4GB", "cpu": "8GB"}
            ))
        ]
        
        # Tester chaque stratégie
        for name, strategy in strategies:
            print(f"\n🔄 Test de la stratégie : {name}")
            try:
                model = strategy()
                print(f"✅ {name} : SUCCÈS")
                
                # Test rapide de génération
                print("🧪 Test de génération...")
                test_input = processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(model.device)
                
                with torch.inference_mode():
                    output = model.generate(
                        **test_input,
                        max_new_tokens=10,
                        do_sample=False
                    )
                
                print("✅ Génération réussie")
                return model, processor, name
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ {name} : ÉCHEC")
                print(f"   Erreur : {error_msg}")
                
                if "disk_offload" in error_msg:
                    print("   → Erreur de disk_offload détectée")
                elif "out of memory" in error_msg.lower():
                    print("   → Erreur de mémoire insuffisante")
                elif "bitsandbytes" in error_msg.lower():
                    print("   → Erreur de bitsandbytes (quantization)")
                
                continue
        
        print("\n❌ Toutes les stratégies ont échoué")
        return None, None, None
        
    except Exception as e:
        print(f"❌ Erreur générale : {e}")
        return None, None, None

def main():
    """Fonction principale"""
    print("🌱 Test de chargement du modèle AgriLens AI")
    print("=" * 50)
    
    # Vérifier les dépendances
    print("📦 Vérification des dépendances...")
    try:
        import transformers
        import accelerate
        print(f"✅ Transformers : {transformers.__version__}")
        print(f"✅ Accelerate : {accelerate.__version__}")
    except ImportError as e:
        print(f"❌ Dépendance manquante : {e}")
        return
    
    # Tester la mémoire
    gpu_memory = test_memory_availability()
    
    # Tester le chargement du modèle
    model, processor, strategy_name = test_model_loading()
    
    if model and processor:
        print(f"\n🎉 SUCCÈS ! Modèle chargé avec la stratégie : {strategy_name}")
        print("✅ L'application devrait fonctionner correctement")
        
        # Nettoyer la mémoire
        del model
        del processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    else:
        print("\n❌ ÉCHEC ! Aucune stratégie n'a fonctionné")
        print("\n💡 Recommandations :")
        print("1. Vérifiez que vous avez suffisamment de mémoire RAM (8GB minimum)")
        print("2. Si vous utilisez Hugging Face Spaces, essayez un runtime avec plus de mémoire")
        print("3. Installez les dépendances : pip install bitsandbytes")
        print("4. Redémarrez l'application")

if __name__ == "__main__":
    main() 
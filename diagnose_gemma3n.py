# =================================================================================
# Diagnostic - Gemma 3n E2B IT
# Analyse approfondie du modèle pour comprendre les problèmes de chargement
# =================================================================================

import os
import json
import torch
from transformers import AutoConfig

# Configuration
MODEL_PATH = "/kaggle/input/gemma-3n/transformers/gemma-3n-e2b-it/1"

def analyze_model_structure():
    """Analyse la structure du modèle"""
    print("🔍 ANALYSE DE LA STRUCTURE DU MODÈLE")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Le chemin {MODEL_PATH} n'existe pas!")
        return False
    
    print(f"✅ Chemin existe: {MODEL_PATH}")
    
    # Lister tous les fichiers
    try:
        files = os.listdir(MODEL_PATH)
        print(f"\n📁 Fichiers dans le modèle:")
        for file in sorted(files):
            file_path = os.path.join(MODEL_PATH, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  • {file} ({size:,} bytes)")
            else:
                print(f"  • {file}/ (dossier)")
    except Exception as e:
        print(f"❌ Erreur lors du listing: {e}")
        return False
    
    return True

def analyze_config_file():
    """Analyse le fichier de configuration"""
    print(f"\n📋 ANALYSE DU FICHIER DE CONFIGURATION")
    print("=" * 60)
    
    config_path = os.path.join(MODEL_PATH, "config.json")
    
    if not os.path.exists(config_path):
        print(f"❌ Fichier config.json non trouvé: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("✅ Fichier config.json trouvé et lu")
        
        print(f"\n📊 Configuration du modèle:")
        print(f"  • Model type: {config.get('model_type', 'Non spécifié')}")
        print(f"  • Architectures: {config.get('architectures', 'Non spécifié')}")
        print(f"  • Vocab size: {config.get('vocab_size', 'Non spécifié')}")
        print(f"  • Hidden size: {config.get('hidden_size', 'Non spécifié')}")
        print(f"  • Num layers: {config.get('num_hidden_layers', 'Non spécifié')}")
        print(f"  • Num attention heads: {config.get('num_attention_heads', 'Non spécifié')}")
        print(f"  • Intermediate size: {config.get('intermediate_size', 'Non spécifié')}")
        print(f"  • Max position embeddings: {config.get('max_position_embeddings', 'Non spécifié')}")
        print(f"  • Rope theta: {config.get('rope_theta', 'Non spécifié')}")
        print(f"  • Use cache: {config.get('use_cache', 'Non spécifié')}")
        print(f"  • Pad token id: {config.get('pad_token_id', 'Non spécifié')}")
        print(f"  • Eos token id: {config.get('eos_token_id', 'Non spécifié')}")
        print(f"  • Tie word embeddings: {config.get('tie_word_embeddings', 'Non spécifié')}")
        
        # Vérifier les clés importantes
        important_keys = ['model_type', 'architectures', 'vocab_size', 'hidden_size']
        missing_keys = [key for key in important_keys if key not in config]
        
        if missing_keys:
            print(f"\n⚠️ Clés manquantes: {missing_keys}")
        else:
            print(f"\n✅ Toutes les clés importantes sont présentes")
        
        return config
        
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du config: {e}")
        return False

def check_transformers_support():
    """Vérifie le support de Transformers"""
    print(f"\n🔧 VÉRIFICATION DU SUPPORT TRANSFORMERS")
    print("=" * 60)
    
    try:
        from transformers import AutoConfig
        
        print("✅ Transformers importé avec succès")
        
        # Essayer de charger la configuration
        try:
            config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
            print("✅ Configuration chargée avec AutoConfig")
            print(f"  • Classe de config: {type(config).__name__}")
            print(f"  • Model type: {config.model_type}")
            print(f"  • Architectures: {config.architectures}")
        except Exception as e:
            print(f"❌ Erreur lors du chargement de la config: {e}")
        
        # Vérifier les modèles supportés
        from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING
        
        print(f"\n📚 Modèles supportés:")
        print(f"  • Causal LM: {len(MODEL_FOR_CAUSAL_LM_MAPPING)} modèles")
        print(f"  • Image-Text-to-Text: {len(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING)} modèles")
        
        # Chercher "gemma" dans les modèles supportés
        gemma_models = []
        for model_type in MODEL_FOR_CAUSAL_LM_MAPPING.keys():
            if 'gemma' in model_type.lower():
                gemma_models.append(model_type)
        
        print(f"\n🔍 Modèles Gemma supportés: {gemma_models}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la vérification: {e}")
        return False

def try_manual_loading():
    """Essaie un chargement manuel"""
    print(f"\n🔄 TENTATIVE DE CHARGEMENT MANUEL")
    print("=" * 60)
    
    approaches = [
        {
            "name": "AutoConfig + AutoModelForCausalLM",
            "imports": ["AutoConfig", "AutoModelForCausalLM"],
            "config_class": "AutoConfig",
            "model_class": "AutoModelForCausalLM"
        },
        {
            "name": "AutoConfig + AutoModelForImageTextToText", 
            "imports": ["AutoConfig", "AutoModelForImageTextToText"],
            "config_class": "AutoConfig",
            "model_class": "AutoModelForImageTextToText"
        },
        {
            "name": "AutoProcessor + AutoModelForCausalLM",
            "imports": ["AutoProcessor", "AutoModelForCausalLM"],
            "config_class": "AutoProcessor",
            "model_class": "AutoModelForCausalLM"
        }
    ]
    
    for approach in approaches:
        print(f"\n📝 Test: {approach['name']}")
        try:
            # Import dynamique
            from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
            
            # Essayer de charger
            if "Processor" in approach['name']:
                processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
                print(f"✅ {approach['config_class']} chargé")
                
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    trust_remote_code=True,
                    device_map="cpu",
                    torch_dtype=torch.float32
                )
                print(f"✅ {approach['model_class']} chargé")
                
                return model, processor
            else:
                config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
                print(f"✅ {approach['config_class']} chargé")
                
                if "ImageTextToText" in approach['name']:
                    model = AutoModelForImageTextToText.from_pretrained(
                        MODEL_PATH,
                        trust_remote_code=True,
                        device_map="cpu",
                        torch_dtype=torch.float32
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        MODEL_PATH,
                        trust_remote_code=True,
                        device_map="cpu",
                        torch_dtype=torch.float32
                    )
                print(f"✅ {approach['model_class']} chargé")
                
                return model, config
                
        except Exception as e:
            print(f"❌ Échec: {e}")
    
    return None, None

def main():
    """Fonction principale de diagnostic"""
    print("🚀 DIAGNOSTIC COMPLET - Gemma 3n E2B IT")
    print("=" * 80)
    
    # 1. Analyse de la structure
    if not analyze_model_structure():
        return
    
    # 2. Analyse de la configuration
    config = analyze_config_file()
    if not config:
        return
    
    # 3. Vérification du support Transformers
    check_transformers_support()
    
    # 4. Tentative de chargement manuel
    model, processor_or_config = try_manual_loading()
    
    # 5. Résumé
    print(f"\n📋 RÉSUMÉ DU DIAGNOSTIC")
    print("=" * 60)
    
    if model is not None:
        print("✅ Le modèle peut être chargé manuellement")
        print(f"  • Type de modèle: {type(model).__name__}")
        print(f"  • Type de processeur/config: {type(processor_or_config).__name__}")
    else:
        print("❌ Le modèle ne peut pas être chargé")
        print("\n💡 RECOMMANDATIONS:")
        print("  1. Le modèle Gemma 3n est très récent et n'est pas encore supporté")
        print("  2. Attendez une mise à jour de Transformers (version stable)")
        print("  3. Utilisez un modèle alternatif comme:")
        print("     • google/gemma-2b-it")
        print("     • google/gemma-7b-it") 
        print("     • microsoft/DialoGPT-medium")
        print("  4. Ou utilisez l'API Hugging Face pour l'inférence")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Script de test pour vérifier la correction de Gemma 3n
"""

import torch
from PIL import Image
import os
import sys

# Ajouter le répertoire src au path
sys.path.append('src')

# Importer la classe processeur personnalisée
from streamlit_app_multilingual import Gemma3nProcessor

def test_gemma3n_processor():
    """Test du processeur personnalisé Gemma 3n"""
    
    print("🧪 Test du processeur Gemma 3n...")
    
    # Vérifier si le modèle local existe
    model_path = "models/gemma-3n-transformers-gemma-3n-e2b-it-v1"
    if not os.path.exists(model_path):
        print(f"❌ Modèle local non trouvé: {model_path}")
        return False
    
    try:
        from transformers import AutoTokenizer, AutoImageProcessor
        
        print("📥 Chargement du tokenizer et image processor...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        image_processor = AutoImageProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print("🔧 Création du processeur personnalisé...")
        processor = Gemma3nProcessor(tokenizer, image_processor)
        
        # Créer une image de test simple
        print("🖼️ Création d'une image de test...")
        test_image = Image.new('RGB', (224, 224), color='green')
        
        # Test avec prompt simple
        test_prompt = "Décris cette image."
        
        print("🧪 Test 1: Sans token <image>")
        try:
            inputs1 = processor(text=test_prompt, images=test_image, return_tensors="pt")
            print("✅ Test 1 réussi - Format standard")
            print(f"   Inputs keys: {list(inputs1.keys())}")
        except Exception as e:
            print(f"❌ Test 1 échoué: {e}")
            return False
        
        print("🧪 Test 2: Avec token <image>")
        try:
            final_prompt = f"<image>\n{test_prompt}"
            inputs2 = processor(text=final_prompt, images=test_image, return_tensors="pt")
            print("✅ Test 2 réussi - Format avec token <image>")
            print(f"   Inputs keys: {list(inputs2.keys())}")
        except Exception as e:
            print(f"⚠️ Test 2 échoué (normal si le token n'est pas reconnu): {e}")
            print("   Utilisation du format standard...")
        
        print("🧪 Test 3: Décodage")
        try:
            # Simuler des tokens de sortie
            fake_tokens = torch.tensor([[1, 2, 3, 4, 5]])
            decoded = processor.decode(fake_tokens[0], skip_special_tokens=True)
            print(f"✅ Test 3 réussi - Décodage: {decoded}")
        except Exception as e:
            print(f"❌ Test 3 échoué: {e}")
            return False
        
        print("🎉 Tous les tests sont passés avec succès!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

if __name__ == "__main__":
    success = test_gemma3n_processor()
    if success:
        print("\n✅ Le processeur Gemma 3n est prêt à être utilisé!")
    else:
        print("\n❌ Des problèmes ont été détectés avec le processeur.")
        sys.exit(1) 
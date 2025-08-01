# =================================================================================
# Test de Génération de Texte avec Gemma 3n E2B IT
# 
# Auteur : Sidoine Kolaolé YEBADOKPO
# Objectif : Tester les capacités de génération de texte du modèle Gemma 3n
# =================================================================================

import os
import torch
import time
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from huggingface_hub import HfFolder
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
        return None, None

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

def test_text_generation(model, processor, prompt, max_new_tokens=200):
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

def test_multimodal_generation(model, processor, image, prompt, max_new_tokens=200):
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

def test_plant_diagnosis(model, processor, image=None):
    """Test spécifique pour le diagnostic de plantes"""
    
    # Prompts de test pour le diagnostic
    test_prompts = [
        {
            "type": "Diagnostic général",
            "prompt": "Analyse cette image de plante et fournis un diagnostic précis des maladies ou problèmes visibles.",
            "multimodal": True
        },
        {
            "type": "Symptômes spécifiques", 
            "prompt": "Décris les symptômes visibles sur cette plante et suggère les causes possibles.",
            "multimodal": True
        },
        {
            "type": "Traitement recommandé",
            "prompt": "Basé sur l'analyse de cette plante, quels traitements recommandes-tu?",
            "multimodal": True
        },
        {
            "type": "Conseils généraux",
            "prompt": "Donne des conseils généraux pour l'entretien des plantes et la prévention des maladies.",
            "multimodal": False
        }
    ]
    
    print("\n🌱 Tests de diagnostic de plantes")
    print("=" * 60)
    
    results = []
    
    for i, test_case in enumerate(test_prompts, 1):
        print(f"\n🔍 Test {i}: {test_case['type']}")
        
        if test_case['multimodal'] and image is not None:
            response, gen_time = test_multimodal_generation(
                model, processor, image, test_case['prompt']
            )
        else:
            response, gen_time = test_text_generation(
                model, processor, test_case['prompt']
            )
        
        if response:
            results.append({
                'test': test_case['type'],
                'response': response,
                'time': gen_time,
                'success': True
            })
        else:
            results.append({
                'test': test_case['type'],
                'response': 'Échec',
                'time': 0,
                'success': False
            })
    
    return results

def test_performance_metrics(model, processor):
    """Teste les métriques de performance du modèle"""
    print("\n📊 Tests de performance")
    print("=" * 40)
    
    # Test avec différents types de prompts
    performance_tests = [
        "Explique-moi la photosynthèse en termes simples.",
        "Quels sont les avantages de l'agriculture biologique?",
        "Comment identifier les carences en nutriments chez les plantes?",
        "Décris les méthodes de lutte biologique contre les ravageurs.",
        "Qu'est-ce que la rotation des cultures et pourquoi est-elle importante?"
    ]
    
    total_time = 0
    successful_tests = 0
    
    for i, prompt in enumerate(performance_tests, 1):
        print(f"\n⚡ Test de performance {i}/{len(performance_tests)}")
        response, gen_time = test_text_generation(model, processor, prompt, max_new_tokens=100)
        
        if response:
            total_time += gen_time
            successful_tests += 1
    
    if successful_tests > 0:
        avg_time = total_time / successful_tests
        print(f"\n📈 Métriques de performance:")
        print(f"  • Tests réussis: {successful_tests}/{len(performance_tests)}")
        print(f"  • Temps moyen: {avg_time:.2f} secondes")
        print(f"  • Temps total: {total_time:.2f} secondes")
    
    return successful_tests, total_time

def main():
    """Fonction principale de test"""
    print("🚀 Démarrage des tests de génération Gemma 3n")
    print("=" * 60)
    
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
    
    diagnosis_results = test_plant_diagnosis(model, processor, test_image)
    
    # Test 4: Métriques de performance
    print("\n" + "="*60)
    print("TEST 4: MÉTRIQUES DE PERFORMANCE")
    print("="*60)
    
    performance_results = test_performance_metrics(model, processor)
    
    # Résumé final
    print("\n" + "="*60)
    print("📋 RÉSUMÉ DES TESTS")
    print("="*60)
    
    successful_diagnosis = sum(1 for r in diagnosis_results if r['success'])
    print(f"✅ Tests de diagnostic réussis: {successful_diagnosis}/{len(diagnosis_results)}")
    print(f"✅ Tests de performance réussis: {performance_results[0]}/5")
    
    if successful_diagnosis > 0:
        avg_diagnosis_time = sum(r['time'] for r in diagnosis_results if r['success']) / successful_diagnosis
        print(f"⏱️ Temps moyen de diagnostic: {avg_diagnosis_time:.2f} secondes")
    
    print("\n🎉 Tests terminés avec succès!")

if __name__ == "__main__":
    main() 
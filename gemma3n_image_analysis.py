# =================================================================================
# ANALYSE D'IMAGE DE PLANTE MALADE AVEC GEMMA 3N
# Auteur : Sidoine Kolaolé YEBADOKPO
# =================================================================================

import torch
import time
from PIL import Image
import os

def load_multimodal_model():
    """Charge le modèle multimodal pour l'analyse d'images"""
    try:
        print("🖼️ Chargement du modèle multimodal...")
        
        from transformers import AutoProcessor, AutoModelForImageTextToText
        
        # Chemin du modèle
        GEMMA_PATH = "/kaggle/input/gemma-3n/transformers/gemma-3n-e2b-it/1"
        
        # Chargement du processeur
        processor = AutoProcessor.from_pretrained(
            GEMMA_PATH,
            local_files_only=True,
            trust_remote_code=True
        )
        
        # Chargement du modèle multimodal (CPU pour économiser la mémoire)
        multimodal_model = AutoModelForImageTextToText.from_pretrained(
            GEMMA_PATH,
            local_files_only=True,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        
        print("✅ Modèle multimodal chargé avec succès")
        return processor, multimodal_model
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle multimodal: {e}")
        return None, None

def analyze_plant_image(image_path: str, processor, model, custom_prompt: str = None):
    """Analyse une image de plante malade"""
    
    try:
        print(f"\n🖼️ Analyse de l'image: {image_path}")
        
        # Vérification de l'existence de l'image
        if not os.path.exists(image_path):
            print(f"❌ Image non trouvée: {image_path}")
            return None
        
        # Chargement et préparation de l'image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))  # Redimensionnement pour économiser la mémoire
        
        print(f"✅ Image chargée: {image.size}")
        
        # Affichage de l'image (optionnel)
        try:
            from IPython.display import display
            display(image)
        except:
            print("📸 Image prête pour analyse")
        
        # Prompt par défaut pour le diagnostic agricole
        if custom_prompt is None:
            prompt = (
                "Analyse cette image de plante. Décris les symptômes visibles et "
                "fournis un diagnostic structuré incluant:\n"
                "1. Nom de la maladie probable\n"
                "2. Symptômes observés (couleur, forme, répartition)\n"
                "3. Causes possibles\n"
                "4. Traitements recommandés\n"
                "5. Mesures préventives pour l'avenir"
            )
        else:
            prompt = custom_prompt
        
        print(f"📝 Prompt d'analyse: {prompt[:100]}...")
        
        # Préparation des inputs
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        
        # Génération du diagnostic
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,  # Plus de tokens pour un diagnostic détaillé
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        generation_time = time.time() - start_time
        
        # Décodage de la réponse
        response = processor.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        # Nettoyage de la réponse
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        print(f"⏱️ Temps d'analyse: {generation_time:.2f}s")
        print(f"📋 Diagnostic généré ({len(response)} caractères)")
        
        return response
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse d'image: {e}")
        return None

def analyze_with_different_prompts(image_path: str, processor, model):
    """Analyse l'image avec différents prompts spécialisés"""
    
    print("\n" + "="*60)
    print("🔬 ANALYSE MULTI-PROMPTS")
    print("="*60)
    
    # Différents prompts pour différents aspects
    prompts = [
        {
            "title": "Diagnostic Général",
            "prompt": "Analyse cette image de plante. Identifie la maladie et décris les symptômes visibles.",
            "max_tokens": 300
        },
        {
            "title": "Symptômes Détaillés",
            "prompt": "Décris en détail les symptômes visibles sur cette plante: couleur des feuilles, forme des taches, répartition des lésions.",
            "max_tokens": 250
        },
        {
            "title": "Traitements Recommandés",
            "prompt": "Basé sur l'image de cette plante malade, recommande des traitements efficaces et des mesures préventives.",
            "max_tokens": 300
        },
        {
            "title": "Diagnostic Expert",
            "prompt": "En tant qu'expert en phytopathologie, analyse cette image et fournis un diagnostic professionnel avec niveau de confiance.",
            "max_tokens": 350
        }
    ]
    
    results = []
    
    for i, prompt_info in enumerate(prompts, 1):
        print(f"\n📋 Analyse {i}: {prompt_info['title']}")
        print("-" * 40)
        
        response = analyze_plant_image(
            image_path=image_path,
            processor=processor,
            model=model,
            custom_prompt=prompt_info['prompt']
        )
        
        if response:
            print("📝 Réponse:")
            print(response)
            results.append({
                "title": prompt_info['title'],
                "response": response
            })
        else:
            print("❌ Échec de l'analyse")
        
        print("-" * 40)
        
        # Pause entre les analyses
        if i < len(prompts):
            print("⏳ Pause de 3 secondes...")
            time.sleep(3)
    
    return results

def main():
    """Fonction principale"""
    
    print("🚀 ANALYSE D'IMAGE DE PLANTE MALADE")
    print("=" * 50)
    
    # Chemin de l'image
    image_path = "/kaggle/input/tomato/tomato_early_blight.jpg"
    
    print(f"📁 Image à analyser: {image_path}")
    
    # Vérification de l'existence de l'image
    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée: {image_path}")
        print("💡 Vérifiez que le dataset tomato est bien attaché à votre notebook")
        return
    
    # Chargement du modèle multimodal
    processor, model = load_multimodal_model()
    
    if processor is None or model is None:
        print("❌ Impossible de charger le modèle multimodal")
        return
    
    # Menu de choix
    print("\n🎯 Choisissez le mode d'analyse:")
    print("1. Analyse simple avec prompt par défaut")
    print("2. Analyse multi-prompts (4 analyses différentes)")
    print("3. Analyse avec prompt personnalisé")
    
    try:
        choice = input("\nVotre choix (1-3): ").strip()
        
        if choice == "1":
            # Analyse simple
            print("\n🔍 Analyse simple...")
            response = analyze_plant_image(image_path, processor, model)
            
            if response:
                print("\n📋 DIAGNOSTIC COMPLET:")
                print("=" * 50)
                print(response)
                print("=" * 50)
        
        elif choice == "2":
            # Analyse multi-prompts
            results = analyze_with_different_prompts(image_path, processor, model)
            
            # Résumé des résultats
            print("\n📊 RÉSUMÉ DES ANALYSES:")
            print("=" * 50)
            for result in results:
                print(f"\n🔍 {result['title']}:")
                print(f"📝 {result['response'][:100]}...")
        
        elif choice == "3":
            # Analyse personnalisée
            custom_prompt = input("\n🌱 Votre prompt personnalisé: ").strip()
            
            if custom_prompt:
                response = analyze_plant_image(image_path, processor, model, custom_prompt)
                
                if response:
                    print("\n📋 RÉSULTAT:")
                    print("=" * 50)
                    print(response)
                    print("=" * 50)
            else:
                print("❌ Prompt vide")
        
        else:
            print("❌ Choix invalide")
            
    except KeyboardInterrupt:
        print("\n👋 Arrêt demandé")
    except Exception as e:
        print(f"❌ Erreur: {e}")
    
    print("\n🎉 Analyse terminée!")

# Exécution directe
if __name__ == "__main__":
    main() 
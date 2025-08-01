# =================================================================================
# GÉNÉRATION DE TEXTE AVEC GEMMA 3N - VERSION OPTIMISÉE
# Auteur : Sidoine Kolaolé YEBADOKPO
# Utilise le modèle déjà chargé: tokenizer et model
# =================================================================================

import torch
import time
from typing import Optional

def monitor_memory(stage: str = ""):
    """Surveille l'utilisation de la mémoire GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        print(f"💾 Mémoire ({stage}): {allocated:.2f} GB / {reserved:.2f} GB")

def generate_text_with_gemma3n(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1
) -> Optional[str]:
    """
    Génère du texte avec le modèle Gemma 3n déjà chargé
    
    Args:
        prompt: Le texte d'entrée
        max_new_tokens: Nombre maximum de tokens à générer
        temperature: Contrôle la créativité (0.0 = déterministe, 1.0 = très créatif)
        top_p: Contrôle la diversité du vocabulaire
        repetition_penalty: Pénalise la répétition
    
    Returns:
        Le texte généré ou None en cas d'erreur
    """
    
    try:
        print(f"\n🔤 Génération de texte avec Gemma 3n...")
        print(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        # Vérification que le modèle est chargé
        if 'tokenizer' not in globals() or 'model' not in globals():
            print("❌ Erreur: Modèle non chargé. Exécutez d'abord le setup.")
            return None
        
        # Configuration du tokenizer si nécessaire
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenisation avec gestion des erreurs
        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,  # Limite pour éviter les dépassements
                padding=True
            )
        except Exception as e:
            print(f"❌ Erreur de tokenisation: {e}")
            return None
        
        # Déplacer vers le bon device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Surveillance mémoire avant génération
        monitor_memory("Avant génération")
        
        # Génération avec timing
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # Paramètres de sécurité
                use_cache=True,
                return_dict_in_generate=False
            )
        
        generation_time = time.time() - start_time
        
        # Surveillance mémoire après génération
        monitor_memory("Après génération")
        
        # Décodage du résultat
        try:
            response = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
        except Exception as e:
            print(f"❌ Erreur de décodage: {e}")
            return None
        
        # Nettoyage de la réponse
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        # Statistiques
        tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        print(f"⏱️ Temps de génération: {generation_time:.2f}s")
        print(f"🔢 Tokens générés: {tokens_generated}")
        print(f"⚡ Vitesse: {tokens_per_second:.1f} tokens/s")
        print(f"📝 Réponse ({len(response)} caractères)")
        
        return response
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {e}")
        return None

def test_agricultural_queries():
    """Test avec des requêtes agricoles spécifiques"""
    
    print("\n" + "="*60)
    print("🧪 TESTS DE GÉNÉRATION AGRICOLE")
    print("="*60)
    
    # Liste des prompts de test
    test_prompts = [
        {
            "title": "Symptômes du mildiou",
            "prompt": "Décris en détail les symptômes de la maladie du mildiou chez les plants de tomate. Inclus l'apparence des feuilles, des tiges et des fruits, ainsi que la progression de la maladie.",
            "max_tokens": 300
        },
        {
            "title": "Méthodes de prévention",
            "prompt": "Explique les méthodes de prévention contre les maladies fongiques dans un jardin potager. Donne des conseils pratiques pour les agriculteurs.",
            "max_tokens": 250
        },
        {
            "title": "Diagnostic rapide",
            "prompt": "Comment identifier rapidement si une plante est malade ? Donne les signes visuels les plus courants à surveiller.",
            "max_tokens": 200
        },
        {
            "title": "Traitements biologiques",
            "prompt": "Quels sont les traitements biologiques efficaces contre les maladies des plantes ? Inclus des recettes naturelles et des méthodes alternatives.",
            "max_tokens": 350
        }
    ]
    
    # Exécution des tests
    for i, test in enumerate(test_prompts, 1):
        print(f"\n📋 Test {i}: {test['title']}")
        print("-" * 40)
        
        response = generate_text_with_gemma3n(
            prompt=test['prompt'],
            max_new_tokens=test['max_tokens'],
            temperature=0.7
        )
        
        if response:
            print("📝 Réponse:")
            print(response)
        else:
            print("❌ Échec de la génération")
        
        print("-" * 40)
        
        # Pause entre les tests pour éviter la surcharge
        if i < len(test_prompts):
            print("⏳ Pause de 2 secondes...")
            time.sleep(2)

def interactive_generation():
    """Mode interactif pour tester des prompts personnalisés"""
    
    print("\n" + "="*60)
    print("🎯 MODE INTERACTIF - GÉNÉRATION PERSONNALISÉE")
    print("="*60)
    print("Tapez 'quit' pour arrêter")
    
    while True:
        try:
            # Saisie utilisateur
            user_prompt = input("\n🌱 Votre prompt: ").strip()
            
            if user_prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Au revoir!")
                break
            
            if not user_prompt:
                print("⚠️ Prompt vide, essayez à nouveau")
                continue
            
            # Paramètres de génération
            try:
                max_tokens = int(input("🔢 Nombre max de tokens (défaut 200): ") or "200")
                temperature = float(input("🌡️ Température (0.1-1.0, défaut 0.7): ") or "0.7")
            except ValueError:
                max_tokens = 200
                temperature = 0.7
                print("⚠️ Valeurs invalides, utilisation des valeurs par défaut")
            
            # Génération
            response = generate_text_with_gemma3n(
                prompt=user_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            
            if response:
                print("\n📝 Réponse de Gemma 3n:")
                print("-" * 30)
                print(response)
                print("-" * 30)
            else:
                print("❌ Échec de la génération")
                
        except KeyboardInterrupt:
            print("\n👋 Arrêt demandé par l'utilisateur")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")

def main():
    """Fonction principale"""
    
    print("🚀 GÉNÉRATION DE TEXTE AVEC GEMMA 3N")
    print("=" * 50)
    
    # Vérification que le modèle est disponible
    if 'tokenizer' not in globals() or 'model' not in globals():
        print("❌ Erreur: Modèle non chargé!")
        print("💡 Exécutez d'abord le setup pour charger le modèle")
        return
    
    print("✅ Modèle Gemma 3n détecté et prêt")
    print(f"📊 Modèle sur device: {next(model.parameters()).device}")
    
    # Menu de choix
    print("\n🎯 Choisissez le mode:")
    print("1. Tests agricoles automatiques")
    print("2. Mode interactif personnalisé")
    print("3. Test simple rapide")
    
    try:
        choice = input("\nVotre choix (1-3): ").strip()
        
        if choice == "1":
            test_agricultural_queries()
        elif choice == "2":
            interactive_generation()
        elif choice == "3":
            # Test simple
            print("\n🧪 Test simple...")
            response = generate_text_with_gemma3n(
                "Explique brièvement l'importance de la rotation des cultures.",
                max_new_tokens=150
            )
            if response:
                print("📝 Réponse:")
                print(response)
        else:
            print("❌ Choix invalide")
            
    except KeyboardInterrupt:
        print("\n👋 Arrêt demandé")
    except Exception as e:
        print(f"❌ Erreur: {e}")
    
    print("\n🎉 Génération terminée!")

# Exécution directe si le script est lancé
if __name__ == "__main__":
    main() 
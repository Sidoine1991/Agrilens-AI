# =================================================================================
# GÉNÉRATION DE TEXTE AVEC GEMMA 3N
# Auteur : Sidoine Kolaolé YEBADOKPO
# =================================================================================

import torch
import time

def generate_text(prompt: str, max_tokens: int = 200):
    """Génère du texte avec le modèle Gemma 3n"""
    
    print(f"🔤 Génération: {prompt[:50]}...")
    
    # Configuration du tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenisation
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    # Déplacer vers le device du modèle
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Génération
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generation_time = time.time() - start_time
    
    # Décodage
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Nettoyage
    if prompt in response:
        response = response.replace(prompt, "").strip()
    
    print(f"⏱️ {generation_time:.2f}s - {len(response)} caractères")
    return response

# =================================================================================
# TESTS DE GÉNÉRATION
# =================================================================================

print("🚀 TESTS DE GÉNÉRATION GEMMA 3N")
print("=" * 50)

# Test 1: Symptômes du mildiou
print("\n📝 Test 1: Symptômes du mildiou")
prompt1 = "Décris les symptômes de la maladie du mildiou chez les plants de tomate."
response1 = generate_text(prompt1, max_tokens=200)
print("📋 Réponse:")
print("-" * 30)
print(response1)
print("-" * 30)

# Test 2: Méthodes de prévention
print("\n📝 Test 2: Méthodes de prévention")
prompt2 = "Explique les méthodes de prévention contre les maladies fongiques dans un jardin potager."
response2 = generate_text(prompt2, max_tokens=250)
print("📋 Réponse:")
print("-" * 30)
print(response2)
print("-" * 30)

# Test 3: Diagnostic rapide
print("\n📝 Test 3: Diagnostic rapide")
prompt3 = "Comment identifier rapidement si une plante est malade ?"
response3 = generate_text(prompt3, max_tokens=150)
print("📋 Réponse:")
print("-" * 30)
print(response3)
print("-" * 30)

print("\n🎉 Tests terminés!") 
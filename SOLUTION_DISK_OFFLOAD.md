# 🔧 Solution pour l'erreur "disk_offload" - Gemma 3n E4B IT

## 🚨 Problème identifié

L'erreur `You are trying to offload the whole model to the disk. Please use the disk_offload function instead.` se produit lorsque :

1. **Le modèle est trop volumineux** pour la mémoire disponible
2. **Hugging Face Spaces** a des limitations de mémoire
3. **Le modèle Gemma 3n E4B IT** nécessite environ 8-12GB de RAM

## ✅ Solution implémentée

### 1. **Stratégies de chargement multiples**

L'application utilise maintenant 4 stratégies de chargement en cascade :

```python
# Stratégie 1: CPU Conservateur
device_map="cpu", torch_dtype=torch.float32, max_memory={"cpu": "8GB"}

# Stratégie 2: 4-bit Quantization  
load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16

# Stratégie 3: 8-bit Quantization
load_in_8bit=True

# Stratégie 4: Gestion mémoire personnalisée
max_memory={0: "4GB", "cpu": "8GB"}
```

### 2. **Dépendances mises à jour**

```bash
pip install bitsandbytes>=0.41.0
pip install accelerate>=0.20.0
pip install transformers>=4.35.0
```

### 3. **Gestion automatique des erreurs**

- Détection automatique de la mémoire disponible
- Fallback automatique entre les stratégies
- Messages d'erreur informatifs

## 🧪 Test de la solution

### Exécuter le script de test :

```bash
python test_model_loading.py
```

Ce script va :
- ✅ Vérifier la mémoire disponible
- ✅ Tester chaque stratégie de chargement
- ✅ Identifier la meilleure stratégie pour votre environnement
- ✅ Fournir des recommandations en cas d'échec

## 🚀 Utilisation

### 1. **Installation des dépendances**

```bash
pip install -r requirements.txt
```

### 2. **Lancement de l'application**

```bash
streamlit run src/streamlit_app_multilingual.py
```

### 3. **Chargement du modèle**

1. Ouvrez l'application dans votre navigateur
2. Allez dans la sidebar "Configuration"
3. Cliquez sur "Charger le modèle Gemma 3n E4B IT"
4. L'application testera automatiquement les stratégies

## 🔍 Diagnostic des problèmes

### Si le chargement échoue :

1. **Vérifiez la mémoire disponible** :
   ```python
   import torch
   if torch.cuda.is_available():
       print(f"GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
   ```

2. **Vérifiez les dépendances** :
   ```bash
   pip list | grep -E "(transformers|accelerate|bitsandbytes)"
   ```

3. **Consultez les logs** :
   - Les messages d'erreur détaillés s'affichent dans l'interface
   - Chaque stratégie testée est documentée

## 💡 Recommandations

### Pour Hugging Face Spaces :

1. **Utilisez un runtime avec plus de mémoire** :
   - CPU: 8GB minimum
   - GPU: 16GB recommandé

2. **Configuration dans `app.py`** :
   ```python
   # Ajoutez ces lignes au début
   import os
   os.environ["TOKENIZERS_PARALLELISM"] = "false"
   os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
   ```

3. **Variables d'environnement** :
   ```bash
   export HF_HOME="/tmp/hf_home"
   export TRANSFORMERS_CACHE="/tmp/transformers_cache"
   ```

### Pour développement local :

1. **Mémoire recommandée** : 16GB RAM minimum
2. **GPU optionnel** : Améliore les performances
3. **Espace disque** : 10GB pour le cache des modèles

## 🛠️ Dépannage avancé

### Erreur "bitsandbytes not found" :

```bash
pip install bitsandbytes --upgrade
# Ou pour CPU uniquement
pip install bitsandbytes-cpu
```

### Erreur "CUDA out of memory" :

1. Réduisez la taille du batch
2. Utilisez la quantification 4-bit
3. Libérez la mémoire GPU :
   ```python
   torch.cuda.empty_cache()
   ```

### Erreur "disk_offload" persistante :

1. Forcez le mode CPU :
   ```python
   device_map="cpu"
   torch_dtype=torch.float32
   ```

2. Utilisez un modèle plus petit :
   ```python
   model_id = "google/gemma-2b-it"  # Au lieu de gemma-3n-E4B-it
   ```

## 📊 Performance attendue

| Stratégie | Mémoire requise | Vitesse | Qualité |
|-----------|----------------|---------|---------|
| CPU Conservateur | 8GB RAM | Lente | Excellente |
| 4-bit Quantization | 4GB RAM | Moyenne | Très bonne |
| 8-bit Quantization | 6GB RAM | Rapide | Bonne |
| Gestion personnalisée | Variable | Variable | Excellente |

## 🔄 Mise à jour automatique

L'application détecte automatiquement :
- ✅ La mémoire disponible
- ✅ Les capacités GPU/CPU
- ✅ Les dépendances installées
- ✅ La meilleure stratégie à utiliser

## 📞 Support

Si le problème persiste :

1. **Exécutez le script de test** et partagez les résultats
2. **Vérifiez les logs** de l'application
3. **Consultez la documentation** Hugging Face
4. **Contactez le support** avec les détails de l'erreur

---

**Note** : Cette solution garantit que l'application fonctionne dans la plupart des environnements, même avec des ressources limitées. 
# 🔧 Guide de Dépannage - Problèmes de Dépendances AgriLens AI

## 🚨 Problème Identifié
Vous rencontrez des conflits de versions entre :
- **PyTorch** : versions incompatibles (2.1.2 vs 2.7.1)
- **torchaudio** : conflit avec PyTorch
- **bitsandbytes** : version incompatible avec PyTorch
- **torchvision** : manquant ou incompatible

## ✅ Solution Complète

### Option 1 : Script Automatique (Recommandé)
Exécutez le script `fix_dependencies.py` que j'ai créé :

```bash
python fix_dependencies.py
```

### Option 2 : Correction Manuelle dans le Notebook

**Remplacez votre cellule problématique par le code du fichier `notebook_fix_cell.py`**

### Option 3 : Commandes Manuelles

```bash
# 1. Nettoyage complet
pip uninstall torch torchaudio torchvision bitsandbytes transformers accelerate -y

# 2. Installation des versions compatibles
pip install torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# 3. BitsAndBytes compatible
pip install bitsandbytes==0.41.1

# 4. Transformers et autres
pip install transformers==4.36.2 accelerate==0.25.0 timm easyocr fastai gitpython
```

## 🔍 Vérification

Après l'installation, vérifiez que tout fonctionne :

```python
import torch
import bitsandbytes
import transformers

print(f"PyTorch: {torch.__version__}")
print(f"BitsAndBytes: {bitsandbytes.__version__}")
print(f"Transformers: {transformers.__version__}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 🎯 Versions Compatibles Testées

| Package | Version | Compatibilité |
|---------|---------|---------------|
| PyTorch | 2.1.2 | ✅ Stable |
| torchaudio | 2.1.2 | ✅ Compatible PyTorch 2.1.2 |
| torchvision | 0.16.2 | ✅ Compatible PyTorch 2.1.2 |
| bitsandbytes | 0.41.1 | ✅ Compatible PyTorch 2.1.2 |
| transformers | 4.36.2 | ✅ Stable |
| accelerate | 0.25.0 | ✅ Compatible |

## 🚀 Prochaines Étapes

1. **Exécutez le script de correction**
2. **Redémarrez le kernel** de votre notebook
3. **Exécutez la nouvelle cellule** de configuration
4. **Testez le chargement du modèle**

## 🔧 Dépannage Supplémentaire

### Si problème de mémoire GPU persiste :
```python
# Utilisez la quantification 4-bit au lieu de 8-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Au lieu de load_in_8bit=True
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
```

### Si le modèle ne se charge toujours pas :
1. Vérifiez que le chemin `GEMMA_PATH` est correct
2. Assurez-vous que tous les fichiers du modèle sont présents
3. Essayez de télécharger le modèle depuis Hugging Face Hub directement

### Si erreurs CUDA :
1. Vérifiez que CUDA est installé
2. Assurez-vous que les drivers GPU sont à jour
3. Vérifiez la compatibilité CUDA avec PyTorch

## 📞 Support

Si les problèmes persistent après avoir suivi ce guide :
1. Vérifiez les logs d'erreur complets
2. Assurez-vous d'avoir suffisamment d'espace disque
3. Redémarrez complètement votre environnement

---

**Note** : Ces versions ont été testées et sont compatibles entre elles. Le problème principal était la mixité de versions PyTorch qui créait des conflits avec bitsandbytes. 
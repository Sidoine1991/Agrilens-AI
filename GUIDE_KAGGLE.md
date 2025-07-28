# 🌱 Guide de Configuration Kaggle pour AgriLens AI

## 📋 Vue d'ensemble

Ce guide vous explique comment configurer Kaggle comme source alternative de modèles pour AgriLens AI, en cas de problèmes de connexion avec Hugging Face.

## 🎯 Avantages de Kaggle

- **Fiabilité** : Infrastructure stable et robuste
- **Modèles optimisés** : Datasets spécialement préparés
- **Alternative gratuite** : Compte gratuit disponible
- **Intégration facile** : API Python simple à utiliser

## 🔧 Configuration étape par étape

### 1. Créer un compte Kaggle

1. Allez sur [kaggle.com](https://www.kaggle.com)
2. Cliquez sur "Register" ou "Sign Up"
3. Créez votre compte gratuit
4. Vérifiez votre email

### 2. Générer une clé API

1. Connectez-vous à votre compte Kaggle
2. Allez dans "Account" (en haut à droite)
3. Faites défiler jusqu'à "API" section
4. Cliquez sur "Create New API Token"
5. Téléchargez le fichier `kaggle.json`

### 3. Configurer l'environnement local

#### Option A : Configuration automatique (recommandée)

1. Lancez l'application avec Kaggle :
   ```bash
   lancer_app_kaggle.bat
   ```

2. Dans l'interface web, allez dans la sidebar "Configuration"
3. Entrez votre nom d'utilisateur Kaggle
4. Entrez votre clé API Kaggle
5. Cliquez sur "Connecter Kaggle"

#### Option B : Configuration manuelle

1. Créez le dossier `.kaggle` dans votre répertoire utilisateur :
   ```bash
   mkdir %USERPROFILE%\.kaggle
   ```

2. Copiez le fichier `kaggle.json` téléchargé dans ce dossier

3. Sur Windows, définissez les permissions :
   ```bash
   icacls "%USERPROFILE%\.kaggle\kaggle.json" /inheritance:r /grant:r "%USERNAME%:F"
   ```

### 4. Installer l'API Kaggle

L'installation se fait automatiquement avec le script `lancer_app_kaggle.bat`, mais vous pouvez aussi l'installer manuellement :

```bash
pip install kaggle
```

## 🚀 Utilisation

### Lancement de l'application

```bash
# Version avec support Kaggle
lancer_app_kaggle.bat

# Ou directement avec Python
streamlit run src/streamlit_app_kaggle.py
```

### Ordre de priorité des sources

L'application essaie les sources dans cet ordre :

1. **🤗 Hugging Face** (par défaut)
   - Modèle : `google/gemma-3n-E4B-it`
   - Timeout : 30-60 secondes

2. **🏆 Kaggle** (fallback)
   - Datasets Kaggle disponibles
   - Téléchargement automatique

3. **💾 Modèle Local** (fallback final)
   - Chemin : `D:/Dev/model_gemma`
   - Fonctionne hors ligne

## 📊 Modèles Kaggle disponibles

### Modèles Gemma sur Kaggle

- `google/gemma-3n-e4b-it` : Modèle principal
- `google/gemma-2b-it` : Modèle plus léger
- `google/gemma-7b-it` : Modèle plus puissant

### Autres modèles de vision

- `plant-disease-detection-models` : Modèles spécialisés
- `agriculture-plant-disease` : Datasets agricoles

## 🔍 Dépannage

### Problème : "Connection timed out"

**Solution :**
1. Vérifiez votre connexion internet
2. Essayez la version Kaggle : `lancer_app_kaggle.bat`
3. Utilisez le modèle local si disponible

### Problème : "Invalid Kaggle credentials"

**Solution :**
1. Vérifiez votre nom d'utilisateur Kaggle
2. Régénérez votre clé API
3. Assurez-vous que le fichier `kaggle.json` est correct

### Problème : "Model not found on Kaggle"

**Solution :**
1. Vérifiez que le dataset existe sur Kaggle
2. Utilisez un dataset alternatif
3. Téléchargez le modèle localement

## 📁 Structure des fichiers

```
AgriLensAI/
├── src/
│   ├── streamlit_app_kaggle.py          # Version avec Kaggle
│   └── streamlit_app_multilingual.py    # Version originale
├── lancer_app_kaggle.bat                # Script de lancement Kaggle
├── models/
│   └── kaggle_gemma/                    # Modèles téléchargés depuis Kaggle
└── GUIDE_KAGGLE.md                      # Ce guide
```

## 🔐 Sécurité

- Ne partagez jamais votre clé API Kaggle
- Le fichier `kaggle.json` contient des informations sensibles
- Utilisez des variables d'environnement en production

## 📞 Support

En cas de problème :

1. Vérifiez ce guide
2. Consultez la documentation Kaggle
3. Vérifiez les logs de l'application
4. Testez avec le modèle local

## 🎉 Avantages de cette solution

✅ **Résout les problèmes de timeout**  
✅ **Alternative fiable à Hugging Face**  
✅ **Configuration simple**  
✅ **Fallback automatique**  
✅ **Gratuit et sécurisé**  

---

*Développé pour la compétition Kaggle sur le diagnostic des maladies de plantes* 
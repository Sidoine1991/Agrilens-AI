# AgriLens AI

## 🇫🇷 Présentation
AgriLens AI est une application de diagnostic de maladies des plantes basée sur le modèle multimodal Gemma 3n de Google. Elle fonctionne **offline**, avec une interface Streamlit bilingue (français/anglais), et est conçue pour la compétition Google - The Gemma 3n Impact Challenge.

- **Interface moderne, responsive et mobile friendly**
- **Bilingue** : tous les textes, boutons, instructions, prompts et résultats sont traduits dynamiquement (français/anglais, sélecteur dans la sidebar)
- **Personnalisation agricole** : choix de la culture (tomate, maïs, manioc, riz, banane, cacao, café, igname, arachide, coton, palmier à huile, ananas, sorgho, mil, patate douce, etc.) et localisation
- **Diagnostic structuré** à partir d'une photo et d'un contexte texte (optionnel)
- **Historique des diagnostics** (consultable, exportable CSV)
- **Mode expert** : prompt complet, annotation/correction, logs
- **Partage facile** : WhatsApp, Facebook, copier, PDF
- **Ressources** : guides PDF, vidéos, contacts d'experts
- **Robuste** : gestion des erreurs, messages clairs, conseils utilisateur

## 🇬🇧 Overview
AgriLens AI is a plant disease diagnosis app powered by Google's Gemma 3n multimodal model. It works **offline**, features a bilingual Streamlit interface (French/English), and is designed for the Google - The Gemma 3n Impact Challenge competition.

- **Modern, responsive, mobile-friendly UI**
- **Bilingual**: all texts, buttons, instructions, prompts and results are dynamically translated (French/English, sidebar selector)
- **Agricultural personalization**: choose crop (tomato, maize, cassava, rice, banana, cocoa, coffee, yam, peanut, cotton, oil palm, pineapple, sorghum, millet, sweet potato, etc.) and location
- **Structured diagnosis** from a photo and optional text context
- **Diagnosis history** (viewable, exportable CSV)
- **Expert mode**: full prompt, annotation/correction, logs
- **Easy sharing**: WhatsApp, Facebook, copy, PDF
- **Resources**: PDF guides, videos, expert contacts
- **Robust**: error handling, clear messages, user tips

---

## 🇫🇷 Installation et lancement (offline)
1. **Cloner ce dépôt et placer les fichiers du modèle Gemma 3n localement** (voir dossier `models/` ou adapter le chemin dans le code).
2. **Créer un environnement virtuel Python** :
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Installer les dépendances** :
   ```powershell
   pip install -r requirements.txt
   ```
4. **Lancer l'application Streamlit** :
   ```powershell
   streamlit run src/streamlit_app.py
   ```
5. **Accéder à l'interface** : [http://localhost:8502](http://localhost:8502)

## 🇬🇧 Installation & Launch (offline)
1. **Clone this repo and place the Gemma 3n model files locally** (see `models/` folder or adjust the path in the code).
2. **Create a Python virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
4. **Launch the Streamlit app**:
   ```powershell
   streamlit run src/streamlit_app.py
   ```
5. **Open the interface**: [http://localhost:8502](http://localhost:8502)

---

## 🇫🇷 Fonctionnalités principales
- **Diagnostic de maladies des plantes** à partir d'une photo et d'un contexte texte (optionnel)
- **Bilingue** : interface, prompts, résultats, boutons, instructions (français/anglais)
- **Personnalisation agricole** : choix de la culture, localisation, prompt adapté
- **Historique des diagnostics** (consultable, exportable CSV)
- **Mode expert** : prompt complet, annotation/correction, logs
- **Partage facile** : WhatsApp, Facebook, copier, PDF
- **Ressources** : guides PDF, vidéos, contacts d'experts
- **Robuste** : gestion des erreurs, interface claire, instructions utilisateur
- **Responsive/mobile** : utilisable sur smartphone/tablette

## 🇬🇧 Main features
- **Plant disease diagnosis** from a photo and optional text context
- **Bilingual**: interface, prompts, results, buttons, instructions (French/English)
- **Agricultural personalization**: crop selection, location, adapted prompt
- **Diagnosis history** (viewable, exportable CSV)
- **Expert mode**: full prompt, annotation/correction, logs
- **Easy sharing**: WhatsApp, Facebook, copy, PDF
- **Resources**: PDF guides, videos, expert contacts
- **Robust**: error handling, clear UI, user instructions
- **Responsive/mobile**: usable on smartphone/tablet

---

## 🇫🇷 Personnalisation et usage terrain
- **Choix de la culture** : menu déroulant (tomate, maïs, manioc, riz, banane, cacao, café, igname, arachide, coton, palmier à huile, ananas, sorgho, mil, patate douce, etc.)
- **Localisation** : champ libre pour préciser la région, le pays, le village
- **Prompt dynamique** : le diagnostic s’adapte à la culture et au contexte local
- **Historique** : chaque diagnostic est sauvegardé, consultable et exportable
- **Mode expert** : annotation/correction, prompt complet, logs
- **Partage** : WhatsApp, Facebook, copier, PDF
- **Ressources** : guides, vidéos, contacts

## 🇬🇧 Personalization & field use
- **Crop selection**: dropdown (tomato, maize, cassava, rice, banana, cocoa, coffee, yam, peanut, cotton, oil palm, pineapple, sorghum, millet, sweet potato, etc.)
- **Location**: free text for region, country, village
- **Dynamic prompt**: diagnosis adapts to crop and local context
- **History**: each diagnosis is saved, viewable, exportable
- **Expert mode**: annotation/correction, full prompt, logs
- **Sharing**: WhatsApp, Facebook, copy, PDF
- **Resources**: guides, videos, contacts

---

## 🇫🇷 Conseils pour la compétition
- L'application fonctionne **100% offline** (aucun appel externe)
- Prévoir plusieurs minutes pour l'inférence sur CPU (modèle volumineux)
- Utiliser les exemples pour tester rapidement l'UI
- Pour la soumission, activer le **mode réel** pour démontrer l'inférence locale

## 🇬🇧 Competition tips
- The app works **100% offline** (no external calls)
- Inference may take several minutes on CPU (large model)
- Use examples for quick UI testing
- For submission, enable **real mode** to demonstrate local inference

---

## 🇫🇷 Temps de chargement du modèle et expérience utilisateur
- **Premier lancement** : le chargement du modèle Gemma 3n peut prendre 1 à 5 minutes selon la puissance de l'ordinateur (RAM, CPU, GPU).
- **Ensuite** : chaque diagnostic (analyse d'image) prend généralement 30 secondes à 2 minutes sur un PC classique (CPU). Sur GPU, c'est plus rapide.
- **Astuce** : laissez l'application ouverte pour éviter de recharger le modèle à chaque fois.
- **Recommandation** : utilisez un PC ou mini-PC avec au moins 16Go de RAM pour un usage fluide.

## 🇬🇧 Model loading time and user experience
- **First launch**: loading the Gemma 3n model may take 1 to 5 minutes depending on your computer's power (RAM, CPU, GPU).
- **Afterwards**: each diagnosis (image analysis) usually takes 30 seconds to 2 minutes on a standard PC (CPU). It's faster with a GPU.
- **Tip**: keep the app running to avoid reloading the model every time.
- **Recommendation**: use a PC or mini-PC with at least 16GB RAM for smooth operation.

---

## Crédits / Credits
- Développé pour le Google - The Gemma 3n Impact Challenge
- Basé sur le modèle Gemma 3n de Google
- Contact : [Votre nom ou équipe] 

---

## 🇫🇷 FAQ / Problèmes connus
### Q : L'application ne se lance pas ou plante au démarrage ?
R : Vérifiez que toutes les dépendances sont installées et que le modèle Gemma 3n est bien présent dans le dossier `models/gemma-3n`.

### Q : L'inférence est très lente !
R : C'est normal sur CPU, le modèle est volumineux. Utilisez les exemples pour tester rapidement l'UI.

### Q : Comment changer la langue de l'interface ?
R : Utilisez le sélecteur de langue dans la barre latérale (sidebar).

### Q : Peut-on utiliser l'application sans connexion Internet ?
R : Oui, tout fonctionne offline si le modèle est bien téléchargé.

### Q : Le mode démo donne toujours la même réponse ?
R : Oui, il sert uniquement à tester l'ergonomie sans attendre l'inférence réelle.

---

## 🇬🇧 FAQ / Known Issues
### Q: The app won't start or crashes at launch?
A: Make sure all dependencies are installed and the Gemma 3n model is present in the `models/gemma-3n` folder.

### Q: Inference is very slow!
A: This is expected on CPU, the model is large. Use examples for quick UI testing.

### Q: How do I change the interface language?
A: Use the language selector in the sidebar.

### Q: Can I use the app without an Internet connection?
A: Yes, everything works offline if the model is downloaded.

### Q: Demo mode always gives the same answer?
A: Yes, it's only for UI testing without waiting for real inference. 

---

## 🇫🇷 Auteur
- **Nom** : Sidoine YEBADOKPO
- **Expertises** : Expert en analyse de données, Développeur Web
- **Téléphone** : +229 01 96 91 13 46
- **Email** : syebadokpo@gmail.com
- [LinkedIn](https://linkedin.com/in/sidoineko)
- [Hugging Face Portfolio](https://huggingface.co/Sidoineko/portfolio)

## 🇬🇧 Author
- **Name**: Sidoine YEBADOKPO
- **Expertise**: Data Analysis Expert, Web Developer
- **Phone**: +229 01 96 91 13 46
- **Email**: syebadokpo@gmail.com
- [LinkedIn](https://linkedin.com/in/sidoineko)
- [Hugging Face Portfolio](https://huggingface.co/Sidoineko/portfolio) 
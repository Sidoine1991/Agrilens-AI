# 🌱 AgriLens AI - Conversion PDF

Ce dossier contient les scripts pour convertir votre note technique Markdown en PDF professionnel.

## 📋 Prérequis

### Windows
- Python 3.8+ installé
- pip (gestionnaire de paquets Python)

### Linux/Mac
- Python 3.8+ installé
- pip3 (gestionnaire de paquets Python)

## 🚀 Utilisation Rapide

### Windows
1. Double-cliquez sur `convert_to_pdf.bat`
2. Le script installera automatiquement les dépendances
3. Le PDF sera généré et ouvert automatiquement

### Linux/Mac
1. Ouvrez un terminal dans ce dossier
2. Exécutez : `chmod +x convert_to_pdf.sh && ./convert_to_pdf.sh`
3. Le PDF sera généré et ouvert automatiquement

## 📁 Fichiers Générés

- **`AgriLens_AI_Technical_Documentation.pdf`** : PDF final professionnel
- **`AgriLens_AI_Technical_Documentation.html`** : Version HTML (si conversion PDF échoue)

## 🔧 Utilisation Manuelle

### Installation des Dépendances
```bash
pip install -r pdf_requirements.txt
```

### Conversion Directe
```bash
python convert_to_pdf_simple.py
```

## 📊 Fonctionnalités du PDF

### Mise en Forme Professionnelle
- **Police moderne** : Inter (Google Fonts)
- **Couleurs cohérentes** : Palette verte AgriLens AI
- **En-tête personnalisé** : Logo et informations du projet
- **Pied de page** : Informations de contact et version

### Optimisations
- **Marges optimisées** : 2cm pour l'impression
- **Sauts de page intelligents** : Évite les coupures dans les tableaux
- **Qualité haute résolution** : 300 DPI
- **Format A4** : Standard international

## 🛠️ Dépannage

### Erreur "WeasyPrint non installé"
```bash
pip install weasyprint
```

### Erreur de police
- Le script utilise des polices web, assurez-vous d'avoir une connexion internet
- En cas de problème, les polices système seront utilisées

### Erreur de conversion PDF
- Le script générera automatiquement un fichier HTML
- Ouvrez-le dans votre navigateur et utilisez Ctrl+P pour imprimer en PDF

## 📝 Personnalisation

### Modifier les Couleurs
Éditez le fichier `convert_to_pdf_simple.py` et modifiez les variables CSS :
```css
--primary-color: #28a745;    /* Vert principal */
--secondary-color: #20c997;  /* Vert secondaire */
--text-color: #333;          /* Couleur du texte */
```

### Modifier la Mise en Page
Ajustez les marges dans la section CSS :
```css
@page {
    size: A4;
    margin: 2cm;  /* Modifiez selon vos besoins */
}
```

## 🔗 Liens Utiles

- **WeasyPrint Documentation** : https://weasyprint.readthedocs.io/
- **Markdown Python** : https://python-markdown.github.io/
- **Google Fonts** : https://fonts.google.com/

## 📞 Support

En cas de problème :
- **Email** : syebadokpo@gmail.com
- **LinkedIn** : https://linkedin.com/in/sidoineko
- **Portfolio** : https://huggingface.co/spaces/Sidoineko/portfolio

---

*Script créé pour AgriLens AI - Technical Documentation Version 3.0* 
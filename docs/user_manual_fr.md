# 🌱 AgriLens AI - Manuel Utilisateur

## 📋 Table des Matières

1. [Introduction](#introduction)
2. [Installation et Configuration](#installation-et-configuration)
3. [Interface Utilisateur](#interface-utilisateur)
4. [Analyse d'Images](#analyse-dimages)
5. [Analyse de Texte](#analyse-de-texte)
6. [Interprétation des Résultats](#interprétation-des-résultats)
7. [Bonnes Pratiques](#bonnes-pratiques)
8. [Dépannage](#dépannage)
9. [Support et Contact](#support-et-contact)
10. [Informations Techniques](#informations-techniques)

---

## 🎯 Introduction

### Qu'est-ce qu'AgriLens AI ?

AgriLens AI est une application innovante de diagnostic des maladies de plantes utilisant l'intelligence artificielle. Développée spécifiquement pour participer à la compétition Kaggle, cette première version représente notre expertise en IA appliquée à l'agriculture.

### Objectifs

- **Diagnostic rapide** : Identifier les maladies de plantes en quelques secondes
- **Conseils pratiques** : Fournir des recommandations d'action concrètes
- **Accessibilité** : Interface simple et intuitive pour tous les agriculteurs
- **Support multilingue** : Disponible en français et anglais

### Public Cible

- Agriculteurs professionnels et amateurs
- Jardiniers et horticulteurs
- Étudiants en agronomie
- Consultants agricoles
- Toute personne intéressée par la santé des plantes

---

## ⚙️ Installation et Configuration

### Prérequis

- Navigateur web moderne (Chrome, Firefox, Safari, Edge)
- Connexion Internet stable
- Compte Hugging Face (pour l'accès aux modèles IA)

### Accès à l'Application

1. **Version en ligne** : Accédez à l'application via Hugging Face Spaces
2. **Version locale** : Clonez le repository et lancez localement

### Configuration Initiale

1. **Sélection de langue** : Choisissez entre français et anglais
2. **Chargement du modèle** : Cliquez sur "Charger le modèle Gemma 2B"
3. **Attendre le chargement** : Le modèle se télécharge automatiquement

---

## 🖥️ Interface Utilisateur

### Structure Générale

L'application est organisée en 4 onglets principaux :

1. **📸 Analyse d'Image** : Diagnostic par photographie
2. **💬 Analyse de Texte** : Diagnostic par description
3. **📖 Manuel Utilisateur** : Guide d'utilisation
4. **ℹ️ À propos** : Informations sur l'application

### Barre Latérale (Configuration)

- **Sélecteur de langue** : Français/English
- **Chargement du modèle** : Bouton pour initialiser l'IA
- **Statut du modèle** : Indicateur de l'état du système

---

## 📸 Analyse d'Images

### Processus d'Analyse

1. **Upload d'image** : Glissez-déposez ou sélectionnez une image
2. **Vérification** : L'application affiche les informations de l'image
3. **Question optionnelle** : Précisez votre préoccupation
4. **Analyse IA** : Le modèle génère un diagnostic
5. **Résultats** : Affichage du diagnostic et des recommandations

### Formats Acceptés

- **PNG** : Format recommandé pour la qualité
- **JPG/JPEG** : Formats courants acceptés
- **Taille minimale** : 500x500 pixels recommandés

### Conseils pour de Meilleurs Résultats

- **Éclairage** : Utilisez un éclairage naturel et uniforme
- **Focus** : Centrez l'image sur la zone malade
- **Résolution** : Utilisez des images de bonne qualité
- **Angles multiples** : Prenez plusieurs photos si nécessaire

### Exemple d'Utilisation

```
1. Photographiez une feuille de tomate avec des taches brunes
2. Uploadez l'image dans l'application
3. Ajoutez la question : "Quelle est cette maladie ?"
4. Obtenez un diagnostic détaillé avec recommandations
```

---

## 💬 Analyse de Texte

### Quand Utiliser l'Analyse de Texte

- Pas d'image disponible
- Description détaillée des symptômes
- Questions générales sur les soins des plantes
- Conseils préventifs

### Structure de Description Recommandée

```
1. Type de plante : Tomate, Laitue, etc.
2. Symptômes observés : Taches, décoloration, etc.
3. Localisation : Feuilles, fruits, tiges, etc.
4. Évolution : Depuis quand, progression
5. Conditions : Arrosage, exposition, température
6. Actions déjà tentées : Traitements appliqués
```

### Exemple de Description

```
"Mes plants de tomates ont des taches brunes circulaires sur les feuilles 
depuis une semaine. Les taches s'agrandissent et certaines feuilles 
jaunissent. J'ai réduit l'arrosage mais ça empire. Les plants sont en 
plein soleil et j'arrose le matin."
```

---

## 🔍 Interprétation des Résultats

### Structure des Résultats

Chaque analyse produit :

1. **Diagnostic** : Identification de la maladie probable
2. **Causes** : Facteurs qui ont pu déclencher le problème
3. **Symptômes** : Description détaillée des signes
4. **Recommandations** : Actions concrètes à entreprendre
5. **Prévention** : Mesures pour éviter la récurrence

### Exemple de Résultat

```
**Diagnostic :** Mildiou de la tomate (Phytophthora infestans)

**Causes possibles :**
• Humidité excessive
• Arrosage sur les feuilles
• Manque de circulation d'air

**Recommandations urgentes :**
• Isolez les plants malades
• Supprimez les feuilles atteintes
• Appliquez un fongicide adapté
• Améliorez la ventilation

**Prévention :**
• Arrosez au pied des plants
• Espacez suffisamment les plants
• Surveillez l'humidité
```

---

## 💡 Bonnes Pratiques

### Pour l'Analyse d'Images

- **Qualité** : Utilisez des images nettes et bien éclairées
- **Cadrage** : Incluez la zone malade et un peu de contexte
- **Échelle** : Prenez des photos à différentes distances
- **Série** : Photographiez l'évolution sur plusieurs jours

### Pour l'Analyse de Texte

- **Précision** : Décrivez les symptômes avec précision
- **Contexte** : Mentionnez les conditions de culture
- **Historique** : Indiquez l'évolution du problème
- **Actions** : Listez les traitements déjà essayés

### Général

- **Régularité** : Surveillez régulièrement vos plants
- **Documentation** : Gardez une trace des diagnostics
- **Consultation** : Consultez un expert pour les cas complexes
- **Prévention** : Appliquez les mesures préventives

---

## 🔧 Dépannage

### Problèmes Courants

#### Erreur de Chargement du Modèle
```
Symptôme : "Modèle non chargé"
Solution : 
1. Vérifiez votre connexion Internet
2. Rechargez la page
3. Cliquez à nouveau sur "Charger le modèle"
```

#### Erreur d'Upload d'Image
```
Symptôme : "Erreur lors de l'upload"
Solution :
1. Vérifiez le format (PNG, JPG, JPEG)
2. Réduisez la taille de l'image
3. Essayez un autre navigateur
```

#### Résultats Imprécis
```
Symptôme : Diagnostic peu fiable
Solution :
1. Améliorez la qualité de l'image
2. Ajoutez une description détaillée
3. Prenez plusieurs photos
4. Consultez un expert pour confirmation
```

### Messages d'Erreur

- **"Modèle non chargé"** : Rechargez le modèle
- **"Erreur d'analyse"** : Vérifiez vos données d'entrée
- **"Timeout"** : Patientez et réessayez
- **"Format non supporté"** : Utilisez PNG, JPG ou JPEG

---

## 📞 Support et Contact

### Créateur de l'Application

**Sidoine Kolaolé YEBADOKPO**
- 📍 **Localisation** : Bohicon, République du Bénin
- 📞 **Téléphone** : +229 01 96 91 13 46
- 📧 **Email** : syebadokpo@gmail.com
- 🔗 **LinkedIn** : linkedin.com/in/sidoineko
- 📁 **Portfolio** : Hugging Face Portfolio: Sidoineko/portfolio

### Version Compétition

Cette première version d'AgriLens AI a été développée spécifiquement pour participer à la compétition Kaggle. Elle représente notre première production publique et démontre notre expertise en IA appliquée à l'agriculture.

### Avertissement Important

⚠️ **Les résultats fournis sont à titre indicatif uniquement. Pour un diagnostic professionnel, consultez un expert qualifié.**

### Comment Obtenir de l'Aide

1. **Documentation** : Consultez ce manuel utilisateur
2. **Interface** : Utilisez l'onglet "À propos" dans l'application
3. **Contact direct** : Utilisez les coordonnées ci-dessus
4. **Communauté** : Rejoignez les forums agricoles

---

## 🔬 Informations Techniques

### Architecture

- **Framework** : Streamlit
- **Modèle IA** : Gemma 2B (Google)
- **Déploiement** : Hugging Face Spaces
- **Langages** : Python, HTML, CSS

### Fonctionnalités Techniques

- **Analyse d'images** : Traitement par IA multimodale
- **Analyse de texte** : Génération de réponses contextuelles
- **Interface responsive** : Adaptée mobile et desktop
- **Support multilingue** : Français et anglais
- **Cache intelligent** : Optimisation des performances

### Sécurité et Confidentialité

- **Données** : Aucune donnée personnelle collectée
- **Images** : Traitées localement, non stockées
- **Modèle** : Exécuté sur serveur sécurisé
- **Connexion** : HTTPS obligatoire

---

## 📚 Ressources Additionnelles

### Documentation Technique

- **Repository GitHub** : Code source complet
- **Documentation API** : Spécifications techniques
- **Guide de déploiement** : Instructions d'installation

### Ressources Agricoles

- **Bases de données** : Référentiels de maladies
- **Guides pratiques** : Méthodes de traitement
- **Communautés** : Forums d'agriculteurs

### Formation et Support

- **Tutoriels vidéo** : Démonstrations pratiques
- **Webinaires** : Sessions de formation
- **Support technique** : Assistance personnalisée

---

## 🎉 Conclusion

AgriLens AI représente une avancée significative dans l'application de l'intelligence artificielle à l'agriculture. Cette première version, développée pour la compétition Kaggle, démontre le potentiel de l'IA pour aider les agriculteurs dans leur travail quotidien.

Nous espérons que cette application vous sera utile et nous vous remercions de votre confiance. N'hésitez pas à nous faire part de vos retours et suggestions pour améliorer les futures versions.

**Bonne utilisation d'AgriLens AI ! 🌱**

---

*Document généré le : [Date]*
*Version : 1.0*
*Créateur : Sidoine Kolaolé YEBADOKPO* 
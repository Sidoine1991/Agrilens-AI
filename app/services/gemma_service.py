"""
Service d'IA pour le diagnostic des maladies des plantes avec Gemma 3n.
Permet l'analyse d'images et de texte pour identifier les maladies des plantes, 
en exploitant le modèle multimodal Gemma 3n avec quantification pour l'optimisation.
"""
import os
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import torch
from PIL import Image
from transformers import (
    AutoProcessor,                  # MODIFIÉ
    AutoModelForImageTextToText,    # MODIFIÉ
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GemmaService:
    """Service pour l'inférence avec le modèle multimodal Gemma 3n-E2B-it."""
    
    def __init__(
        self,
        model_name: str = "google/gemma-3n-E2B-it",
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
    
    def load_model(self):
        """Charge le pipeline image-text-to-text."""
        if self.model is not None:
            logger.info("Le modèle est déjà chargé.")
            return
        try:
            logger.info(f"Chargement du pipeline pour {self.model_name}...")
            self.model = pipeline(
                "image-text-to-text",
                model=self.model_name,
                cache_dir=self.cache_dir
            )
            logger.info("Modèle chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}", exc_info=True)
            raise

    def diagnose_plant_disease(
        self,
        image: Optional[Image.Image] = None,
        prompt_text: str = "",
        generation_options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Effectue un diagnostic de maladie des plantes à partir d'une image et/ou d'un prompt textuel.
        """
        if self.model is None:
            raise RuntimeError("Le modèle doit être chargé avant de pouvoir faire un diagnostic. Appelez .load_model()")
        if image is None and not prompt_text:
            raise ValueError("Au moins une image ou un prompt textuel doit être fourni.")
        try:
            content = []
            if image:
                content.append({"type": "image", "image": image.convert("RGB")})
            if prompt_text:
                content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            kwargs = generation_options or {}
            response = self.model(text=messages, **kwargs)
            return response[0]["generated_text"] if response and "generated_text" in response[0] else str(response)
        except Exception as e:
            logger.error(f"Erreur lors du diagnostic: {str(e)}", exc_info=True)
            raise

# --- Utilisation du service ---

# "Singleton" pour le service, pour ne le charger qu'une fois
gemma_service_instance: Optional[GemmaService] = None

def get_gemma_service() -> GemmaService:
    """Retourne une instance unique du service Gemma, en le chargeant si nécessaire."""
    global gemma_service_instance
    if gemma_service_instance is None:
        logger.info("Création d'une nouvelle instance de GemmaService.")
        gemma_service_instance = GemmaService()
        gemma_service_instance.load_model()
    return gemma_service_instance


# Exemple d'utilisation (peut être testé en exécutant ce fichier directement)
if __name__ == "__main__":
    try:
        # Création et chargement du service
        gemma_service = get_gemma_service()
        
        # --- Exemple 1: Diagnostic avec image et prompt ---
        # NOTE : Remplacez 'path/to/your/image.jpg' par un vrai chemin
        try:
            image_path = "path/to/your/image.jpg" # Remplacez par le chemin de votre image test
            test_image = Image.open(image_path)
            
            prompt = (
                "Veuillez analyser les symptômes visuels sur cette feuille de plante avec soin et structurer votre réponse comme suit :\n\n"
                "**1. Description Détaillée des Symptômes Visuels** (Concentrez-vous *exclusivement* sur ce que vous observez : couleur, forme, texture, etc., SANS immédiatement identifier la maladie.)\n\n"
                "**2. Diagnostic Possible et Explication**\n"
            )
            
            print("\n--- Diagnostic avec image et prompt ---")
            diagnosis_result = gemma_service.diagnose_plant_disease(
                image=test_image,
                prompt_text=prompt
            )
            print(diagnosis_result)

        except FileNotFoundError:
            print(f"\nAVERTISSEMENT: L'image de test sur '{image_path}' n'a pas été trouvée. Le test d'image est sauté.")
        
        # --- Exemple 2: Question textuelle uniquement ---
        print("\n--- Question textuelle uniquement ---")
        text_prompt = "Quels sont les traitements biologiques contre le mildiou de la tomate ?"
        text_result = gemma_service.diagnose_plant_disease(prompt_text=text_prompt)
        print(text_result)
        
    except Exception as e:
        print(f"Une erreur est survenue lors de l'exécution de l'exemple: {str(e)}")

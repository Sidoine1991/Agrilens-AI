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
    GenerationConfig
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GemmaService:
    """Service pour l'inférence avec le modèle multimodal Gemma 3n."""
    
    def __init__(
        self,
        model_name: str = "google/gemma-3n-E4B-it-litert-preview", # MODÈLE SPÉCIFIQUE
        cache_dir: Optional[Union[str, Path]] = None,
        use_quantization: bool = True
    ):
        """
        Initialise le service Gemma 3n.
        
        Args:
            model_name: Nom du modèle Gemma à utiliser.
            cache_dir: Dossier de cache pour les modèles.
            use_quantization: Activer la quantification 4-bit pour le chargement du modèle sur GPU.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.use_quantization = use_quantization
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        
        logger.info(f"Initialisation du service Gemma 3n sur le périphérique: {self.device}")
    
    def load_model(self):
        """Charge le modèle et le processeur multimodal."""
        if self.model is not None:
            logger.info("Le modèle est déjà chargé.")
            return

        try:
            logger.info(f"Chargement du processeur multimodal pour {self.model_name}...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            logger.info(f"Chargement du modèle {self.model_name}...")
            
            quantization_config = None
            if self.use_quantization and self.device.type == 'cuda':
                logger.info("Utilisation de la quantification 4-bit (BitsAndBytes).")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float32,
                    bnb_4bit_use_double_quant=True,
                )
            
            # Chargement du modèle multimodal
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto" if self.device.type == 'cuda' else None, # Gère le placement sur GPU
                cache_dir=self.cache_dir
            )

            # Si on est sur CPU, il faut explicitement déplacer le modèle
            if self.device.type == 'cpu':
                self.model.to(self.device)

            logger.info("Modèle et processeur chargés avec succès")
            
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
        
        Args:
            image: Objet PIL.Image de la plante malade.
            prompt_text: Description textuelle des symptômes ou question.
            generation_options: Dictionnaire d'options pour la génération de texte (ex: max_new_tokens).
            
        Returns:
            Le texte généré par le modèle.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Le modèle doit être chargé avant de pouvoir faire un diagnostic. Appelez .load_model()")
            
        if image is None and not prompt_text:
            raise ValueError("Au moins une image ou un prompt textuel doit être fourni.")
        
        try:
            # Construction du message multimodal
            content = []
            if image:
                content.append({"type": "image", "image": image.convert("RGB")})
            if prompt_text:
                content.append({"type": "text", "text": prompt_text})

            messages = [{"role": "user", "content": content}]
            
            # Préparation des inputs pour le modèle
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt"
            ).to(self.model.device) # S'assurer que les inputs sont sur le même device que le modèle
            
            # Configuration de la génération
            default_gen_options = {
                'max_new_tokens': 512,
                'do_sample': True,
                'temperature': 0.7,
            }
            if generation_options:
                default_gen_options.update(generation_options)

            generation_config = GenerationConfig(**default_gen_options)
            
            # Génération de la réponse
            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=generation_config)
            
            # Décodage de la réponse
            input_len = inputs["input_ids"].shape[-1]
            response = self.processor.batch_decode(
                outputs[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            
            return response
            
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

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from typing import Optional

class VisionCaptionService:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

    def generate_caption(self, image: Image.Image, prompt: Optional[str] = None, max_length: int = 100) -> str:
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_length=max_length)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

# Singleton pour le service
vision_caption_service_instance: Optional[VisionCaptionService] = None

def get_vision_caption_service() -> VisionCaptionService:
    global vision_caption_service_instance
    if vision_caption_service_instance is None:
        vision_caption_service_instance = VisionCaptionService()
    return vision_caption_service_instance

# Exemple d'utilisation
if __name__ == "__main__":
    service = get_vision_caption_service()
    img = Image.open("path/to/your/image.jpg")
    print(service.generate_caption(img)) 
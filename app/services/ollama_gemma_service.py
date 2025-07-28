import requests
from typing import Optional

class OllamaGemmaService:
    def __init__(self, model: str = "gemma3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def query(self, prompt: str, stream: bool = False) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")

# Singleton pour le service
ollama_gemma_service_instance: Optional[OllamaGemmaService] = None

def get_ollama_gemma_service() -> OllamaGemmaService:
    global ollama_gemma_service_instance
    if ollama_gemma_service_instance is None:
        ollama_gemma_service_instance = OllamaGemmaService()
    return ollama_gemma_service_instance

# Exemple d'utilisation
if __name__ == "__main__":
    service = get_ollama_gemma_service()
    prompt = "Bonjour, peux-tu me donner un conseil pour la culture de la tomate ?"
    print(service.query(prompt)) 
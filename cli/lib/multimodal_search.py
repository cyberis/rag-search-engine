import os
from typing import TypedDict
from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch():
    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str):
        return self.model.encode(text)

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        return self.model.encode(image)
    
def verify_image_embedding(image_path: str) -> None:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    try:
        image = Image.open(image_path)
        image.verify()  # Verify that it's a valid image
    except Exception as e:
        raise ValueError(f"Invalid image file: {image_path}. Error: {e}")
    
    # Verify that the image can be encoded by the model
    encoder = MultimodalSearch()
    image_embedding = encoder.embed_image(image_path)
    print(f"Embedding shape: {image_embedding.shape[0]} dimensions")
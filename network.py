import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClipNetwork:
    """
    Network which performs transformation of images and texts to latent space
    """

    def __init__(self, model_config="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_config).to(device)  # hugging lover
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_config)

    def transform_images(self, images):
        inputs = self.processor(images=images, return_tensors="pt", padding=True, device=device)
        inputs['pixel_values'] = inputs['pixel_values'].to(device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        return outputs.to('cpu').numpy()

    def transform_texts(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

        return outputs.to('cpu').numpy()

    def find_top_k(self, query, embeddings, top_k):
        inputs = self.processor(text=[query], return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)

        with torch.no_grad():
            query_embedding = self.model.get_text_features(**inputs)

        query_embedding = query_embedding.cpu().numpy()
        embeddings = np.array(embeddings)

        similarities = np.dot(embeddings, query_embedding.T).flatten()
        similarities = similarities / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))

        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        return top_k_indices

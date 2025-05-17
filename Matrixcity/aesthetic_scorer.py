import os
import torch
import open_clip
import torch.nn as nn
from urllib.request import urlretrieve

class AestheticScorer:
    def __init__(self, ckpt_path="checkpoint/ckpt", device=None):
        self.ckpt_path = ckpt_path
        os.makedirs(self.ckpt_path, exist_ok=True)
        
        self.aesthetic_model_url = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
        )
        self.aesthetic_model_ckpt_path = os.path.join(self.ckpt_path, "sa_0_4_vit_l_14_linear.pth")
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self._download_aesthetic_model()
        self._load_models()
        
    def _download_aesthetic_model(self):
        if not os.path.exists(self.aesthetic_model_ckpt_path):
            print("Downloading aesthetic model...")
            urlretrieve(self.aesthetic_model_url, self.aesthetic_model_ckpt_path)
            print(f"Aesthetic model downloaded to {self.aesthetic_model_ckpt_path}")
        else:
            print("Aesthetic model already exists. Skipping download.")
    
    def _load_models(self):        
        # Initialize and load the aesthetic predictor
        self.aesthetic_model = nn.Linear(768, 1)
        checkpoint = torch.load(self.aesthetic_model_ckpt_path, map_location=self.device)
        self.aesthetic_model.load_state_dict(checkpoint)
        self.aesthetic_model = self.aesthetic_model.to(self.device)
        self.aesthetic_model.eval()
        
        # Load the CLIP model and preprocessor
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        
        print("Models loaded successfully.")
    
    def score_image(self, image):
        # Load and preprocess the image
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract CLIP features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Predict the aesthetic score
        with torch.no_grad():
            aesthetic_score = self.aesthetic_model(image_features).item()
        
        return aesthetic_score
    
    def score_images(self, images):
        """
        Batch scoring for multiple images.
        Args:
        - images (list of PIL.Image): A list of images to score.
        Returns:
        - scores (list of float): A list of aesthetic scores for each image.
        """
        # Preprocess all images in the batch
        image_inputs = torch.stack([self.clip_preprocess(image) for image in images]).to(self.device)
        # Extract CLIP features for the batch
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        # Predict aesthetic scores for the batch
        with torch.no_grad():
            aesthetic_scores = self.aesthetic_model(image_features).squeeze(-1).tolist()
        return aesthetic_scores

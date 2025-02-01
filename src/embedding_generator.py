import os
import cv2
from io import BytesIO
import logging
from torch.hub import download_url_to_file

import torch
import numpy as np
from PIL import Image
from imagebind import data
from imagebind.models import imagebind_model

from torchvision import transforms


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Gera embeddings multimodais usando ImageBind"""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.model = self._load_model()
        
    def _load_model(self):
        """Initialize and test the ImageBind model."""
        checkpoint_path = "~/.cache/torch/checkpoints/imagebind_huge.pth"
        os.makedirs(os.path.expanduser("~/.cache/torch/checkpoints"), exist_ok=True)

        if not os.path.exists(os.path.expanduser(checkpoint_path)):
            print("Baixando pesos do ImageBind...")
            download_url_to_file(
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                os.path.expanduser(checkpoint_path)
            )
            
        try:
            
            checkpoint_path = os.path.expanduser("~/.cache/torch/checkpoints/imagebind_huge.pth")
        
            # Verifique se o arquivo existe
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
                
            model = imagebind_model.imagebind_huge(pretrained=False)
            model.load_state_dict(torch.load(checkpoint_path))
            model.eval().to(self.device)
            
            # model = imagebind_model.imagebind_huge(pretrained=True)
            # model.eval()
            # model.to(self.device)

            # Quick test with empty text input
            logger.info("Testing model with sample input...")
            test_input = data.load_and_transform_text([""], self.device)
            with torch.no_grad():
                _ = model({"text": test_input})
            
            logger.info("🤖 ImageBind model initialized successfully")
            return model
        except Exception as e:
            logger.error(f"🚨 Model initialization failed: {str(e)}")
            raise

    
    def generate_embedding_old(self, input_data, modality):
        """Gera embedding para diferentes modalidades"""
        processors = {
            "vision": self.process_vision,
            "audio": self.process_audio,
            "text": self.process_text,
            "depth": self.process_depth
        }
        
        if modality not in processors:
            raise ValueError(f"Modalidade não suportada: {modality}")
            
        inputs = {modality: processors[modality](input_data)}
        
        with torch.no_grad():
            embedding = self.model(inputs)[modality]
        return embedding.squeeze(0).cpu().numpy()
    
    def generate_embedding(self, input_data, modality):
        """Gera embedding para diferentes modalidades"""
        processors = {
            "vision": lambda x: data.load_and_transform_vision_data(x, self.device),
            "audio": lambda x: data.load_and_transform_audio_data(x, self.device),
            "text": lambda x: data.load_and_transform_text(x, self.device),
            "depth": self.process_depth
        }
        
        try:
            # Verificação de tipo de entrada
            if not isinstance(input_data, list):
                raise ValueError(f"Input data must be a list. Received: {type(input_data)}")
                
            inputs = {modality: processors[modality](input_data)}
            with torch.no_grad():
                embedding = self.model(inputs)[modality]
            return embedding.squeeze(0).cpu().numpy()
        except Exception as e:
            logger.error(f"Error generating {modality} embedding: {str(e)}", exc_info=True)
            raise
    

    def process_vision(self, image_path):
        """Processa imagem"""
        return data.load_and_transform_vision_data([image_path], self.device)
    
    def process_audio(self, audio_path):
        """Processa áudio"""
        return data.load_and_transform_audio_data([audio_path], self.device)
    
    def process_text(self, text):
        """Processa texto"""
        return data.load_and_transform_text([text], self.device)
    
    def process_depth(self, depth_paths, device="cpu"):
        """Processamento customizado para depth maps"""
        try:
            print("p1:",depth_paths)
            # Verificar existencia dos arquivos
            for path in depth_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Arquivo de depth map não encontrado: {path}")
            
            # Carregar e transformar
            depth_images = [Image.open(path).convert("L") for path in depth_paths]
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            
            return torch.stack([transform(img) for img in depth_images]).to(device)
            
        except Exception as e:
            logger.error(f"🚨 Erro no processamento de depth map: {str(e)}")
            raise
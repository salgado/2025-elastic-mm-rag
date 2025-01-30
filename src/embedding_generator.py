# src/embedding_generator.py

import os
import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import logging
from torch.hub import download_url_to_file


from imagebind import data
from imagebind.models import imagebind_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, device=None):
        """Initialize the EmbeddingGenerator with specified device."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.model = self._initialize_model()

    def _initialize_model(self):
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

    def process_modality(self, content_path, modality):
        """Process various modalities including video.
        
        Args:
            content_path: Path to content file or text string
            modality: One of ["vision", "audio", "text", "video", "depth"]
            
        Returns:
            numpy.ndarray: Embedding vector for the content
        """
        if modality not in ["vision", "audio", "text", "video", "depth"]:
            logger.warning(f"⚠️ Unsupported modality: '{modality}'")
            return None

        try:
            if modality == "video":
                return self._process_video(content_path)
            else:
                return self._process_with_imagebind(content_path, modality)
        except Exception as e:
            logger.error(f"⚡ Error processing {modality}: {str(e)}")
            return None

    def _process_with_imagebind(self, content_path, modality):
        """Process content using ImageBind's built-in processors."""
        processors = {
            "vision": data.load_and_transform_vision_data,
            "audio": data.load_and_transform_audio_data,
            "text": data.load_and_transform_text,
            "depth": data.load_and_transform_vision_data
        }

        # Some modalities need CPU processing
        processing_device = torch.device("cpu") if modality in ["audio", "vision", "depth"] else self.device
        self.model = self.model.to(processing_device)

        # Process the input
        inputs = {modality: processors[modality]([content_path], processing_device)}
        with torch.no_grad():
            embeddings = self.model(inputs)
        return embeddings[modality].cpu().numpy()[0]

    def _process_video(self, video_path, num_frames=8):
        """Process video by extracting frames and computing embeddings."""
        if not os.path.exists(video_path):
            logger.warning(f"⚠️ Video not found: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.warning("⚠️ Could not read frames from video.")
            return None

        # Extract evenly spaced frames
        frames_to_extract = min(num_frames, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, frames_to_extract).astype(int)
        frame_embeddings = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            emb = self._frame_to_embedding(frame_rgb)
            if emb is not None:
                frame_embeddings.append(emb)

        cap.release()

        if not frame_embeddings:
            logger.warning("⚠️ Could not extract video embeddings.")
            return None

        return np.mean(frame_embeddings, axis=0)

    def _frame_to_embedding(self, rgb_frame):
        """Convert a single video frame to embedding."""
        pil_image = Image.fromarray(rgb_frame)
        temp_buffer = BytesIO()
        pil_image.save(temp_buffer, format='PNG')
        temp_buffer.seek(0)

        # Save temporary frame
        temp_frame_path = "temp_frame.png"
        with open(temp_frame_path, "wb") as f:
            f.write(temp_buffer.getvalue())

        # Process frame
        emb = self._process_with_imagebind(temp_frame_path, "vision")

        # Cleanup
        try:
            os.remove(temp_frame_path)
        except Exception as e:
            logger.warning(f"Could not remove temporary frame: {e}")

        return emb

# Test function
def test_embedding_generator():
    """Test the EmbeddingGenerator with various modalities."""
    generator = EmbeddingGenerator()

    # Test text embedding
    text = "A dark night in Gotham City"
    text_emb = generator.process_modality(text, "text")
    logger.info(f"Text embedding shape: {text_emb.shape if text_emb is not None else 'Failed'}")

    # Test image embedding if sample image exists
    image_path = "data/images/crime_scene_1.jpg"
    if os.path.exists(image_path):
        image_emb = generator.process_modality(image_path, "vision")
        logger.info(f"Image embedding shape: {image_emb.shape if image_emb is not None else 'Failed'}")

if __name__ == "__main__":
    test_embedding_generator()
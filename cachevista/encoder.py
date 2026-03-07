import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

_instances: dict[str, "CLIPEncoder"] = {}
_sent_model: SentenceTransformer | None = None


def get_encoder(model_name: str = "openai/clip-vit-large-patch14") -> "CLIPEncoder":
    if model_name not in _instances:
        _instances[model_name] = CLIPEncoder(model_name)
    return _instances[model_name]


def get_sentence_encoder() -> SentenceTransformer:
    global _sent_model
    if _sent_model is None:
        print("loading sentence encoder (all-MiniLM-L6-v2)...")
        _sent_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("sentence encoder loaded")
    return _sent_model


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CLIPEncoder:
    """
    Vision-only CLIP encoder with joint (image, question) embedding support.

    encode(image) -> 768-dim unit-norm image embedding
    encode_joint(image, question) -> 1152-dim unit-norm joint embedding
        = L2_norm(concat(clip_image_emb(768), sentence_emb(384)))

    The joint embedding is what CacheVista uses for L2 FAISS search and L3 MLP.
    L1 hash is computed from hash(image_bytes + question.encode()).

    Contract: all returned embeddings are unit-norm float32.
    Use encoder.image_dim (768) and encoder.joint_dim (1152), not hardcoded values.
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        torch.manual_seed(42)
        np.random.seed(42)

        print(f"loading {model_name} (vision only)...")
        self.model_name = model_name
        self.device = get_device()
        print(f"using device: {self.device}")

        if self.device.type == "cuda":
            torch.cuda.manual_seed(42)

        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.image_dim = self.model.config.projection_dim
        self.sent_dim = 384  # all-MiniLM-L6-v2
        self.joint_dim = self.image_dim + self.sent_dim

        print(f"CLIP loaded — image_dim={self.image_dim}  joint_dim={self.joint_dim}  "
              f"resize={self.processor.size}  normalize={self.processor.do_normalize}")

        self._warmup()

    def __repr__(self):
        return f"CLIPEncoder(model={self.model_name}, image_dim={self.image_dim}, joint_dim={self.joint_dim})"

    def _warmup(self):
        dummy = Image.new("RGB", (224, 224), color=128)
        self.encode(dummy)
        self.encode_joint(dummy, "warmup question")

    def encode(self, image: Image.Image) -> np.ndarray:
        if not isinstance(image, Image.Image):
            raise TypeError(f"expected PIL.Image, got {type(image)}")
        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError(f"image has zero dimension: {image.size}")

        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self.model(**inputs).image_embeds

        emb = features[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def encode_joint(self, image: Image.Image, question: str) -> np.ndarray:
        """
        Joint (image, question) embedding.
        Concatenates L2-normalized CLIP image emb with sentence emb, then re-normalizes.
        This is the embedding used for cache lookup and drift detection.
        """
        img_emb = self.encode(image)
        sent_emb = get_sentence_encoder().encode(
            question, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

        joint = np.concatenate([img_emb, sent_emb])
        norm = np.linalg.norm(joint)
        return joint / norm if norm > 0 else joint

    def encode_batch(self, images: list[Image.Image], chunk_size: int = 32) -> np.ndarray:
        """Image-only batch encoding. Returns (N, image_dim) normalized array."""
        if not images:
            return np.empty((0, self.image_dim), dtype=np.float32)

        results = []
        for i in range(0, len(images), chunk_size):
            chunk = [img.convert("RGB") for img in images[i:i + chunk_size]]
            inputs = self.processor(images=chunk, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = self.model(**inputs).image_embeds

            embs = features.cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            results.append(embs / norms)

        return np.concatenate(results, axis=0)

    def encode_joint_batch(self, images: list[Image.Image],
                           questions: list[str]) -> np.ndarray:
        """
        Joint batch encoding for (image, question) pairs.
        Returns (N, joint_dim) normalized array.
        """
        assert len(images) == len(questions), "images and questions must be same length"
        if not images:
            return np.empty((0, self.joint_dim), dtype=np.float32)

        img_embs = self.encode_batch(images)
        sent_embs = get_sentence_encoder().encode(
            questions, normalize_embeddings=True,
            batch_size=64, show_progress_bar=False
        ).astype(np.float32)

        joints = np.concatenate([img_embs, sent_embs], axis=1)
        norms = np.linalg.norm(joints, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return joints / norms
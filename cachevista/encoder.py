import logging
import threading
import warnings

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from cachevista.utils import get_device

logger = logging.getLogger(__name__)

_instances: dict[str, "CLIPEncoder"] = {}
_instances_lock = threading.Lock()

_sent_model: SentenceTransformer | None = None
_sent_model_lock = threading.Lock()


def get_encoder(model_name: str = "openai/clip-vit-large-patch14") -> "CLIPEncoder":
    if model_name in _instances:
        return _instances[model_name]
    with _instances_lock:
        if model_name not in _instances:
            _instances[model_name] = CLIPEncoder(model_name)
    return _instances[model_name]


def get_sentence_encoder() -> SentenceTransformer:
    global _sent_model
    if _sent_model is not None:
        return _sent_model
    with _sent_model_lock:
        if _sent_model is None:
            logger.info("loading sentence encoder (all-MiniLM-L6-v2)...")
            _sent_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("sentence encoder loaded")
    return _sent_model


class CLIPEncoder:
    """
    Vision-only CLIP encoder with joint (image, question) embedding support.

    encode(image) -> 768-dim unit-norm image embedding
    encode_joint(image, question) -> 1152-dim unit-norm joint embedding
        = L2_norm( concat( clip_image_emb(768), sentence_emb(384) ) )
        i.e. concat first, then normalize — NOT normalize-each-then-concat

    Joint embedding design note: both component vectors are unit-norm before
    concatenation. After concat, re-normalizing divides by sqrt(2) (a constant
    since both inputs are unit vectors). The image component (768-dim) contributes
    ~66.7% of raw magnitude vs ~33.3% for the text component (384-dim), so cosine
    similarity in FAISS will be biased toward image similarity. This is an explicit
    design choice — no learned projection or dimensionality alignment is applied.

    Contract: all returned embeddings are unit-norm float32.
    Use encoder.image_dim (768) and encoder.joint_dim (1152), not hardcoded values.

    Benchmark note: encoding cost (one CLIP forward pass + one sentence transformer
    forward pass) is not included in cache lookup latency benchmarks. Reported
    sub-millisecond cache latency measures only the FAISS search step, not end-to-end
    query latency. In a real VLM pipeline, encoding happens before the cache lookup
    and dominates total latency when the cache misses.
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        logger.info(f"loading {model_name} (vision only)...")
        self.model_name = model_name
        self.device = get_device()
        logger.info(f"using device: {self.device}")

        dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32

        self.model = CLIPVisionModelWithProjection.from_pretrained(
            model_name, torch_dtype=dtype
        )
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.image_dim = self.model.config.projection_dim
        self.sent_dim = 384  # all-MiniLM-L6-v2
        self.joint_dim = self.image_dim + self.sent_dim

        logger.info(
            f"CLIP loaded — image_dim={self.image_dim}  joint_dim={self.joint_dim}  "
            f"resize={self.processor.size}  normalize={self.processor.do_normalize}"
        )

        self._warmup()

    def __repr__(self):
        return f"CLIPEncoder(model={self.model_name}, image_dim={self.image_dim}, joint_dim={self.joint_dim})"

    def _warmup(self):
        dummy = Image.new("RGB", (224, 224), color=128)
        emb = self.encode(dummy)
        joint_emb = self.encode_joint(dummy, "warmup question")

        if emb.shape != (self.image_dim,):
            raise RuntimeError(
                f"encode() output shape {emb.shape} != expected ({self.image_dim},) — "
                "model projection_dim may not match the expected architecture"
            )
        if joint_emb.shape != (self.joint_dim,):
            raise RuntimeError(
                f"encode_joint() output shape {joint_emb.shape} != expected ({self.joint_dim},)"
            )

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(emb)
        if norm == 0:
            warnings.warn(
                "zero-norm embedding encountered — returning unnormalized vector, "
                "which breaks the unit-norm contract expected by downstream FAISS operations",
                RuntimeWarning,
                stacklevel=2,
            )
            return emb
        return emb / norm

    def _normalize_batch(self, embs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        zero_rows = (norms.squeeze(axis=1) == 0)
        if zero_rows.any():
            warnings.warn(
                f"{zero_rows.sum()} zero-norm embeddings in batch — those rows "
                "will not be normalized, breaking the unit-norm contract",
                RuntimeWarning,
                stacklevel=2,
            )
        norms = np.where(norms > 0, norms, 1.0)
        return embs / norms

    def encode(self, image: Image.Image) -> np.ndarray:
        if not isinstance(image, Image.Image):
            raise TypeError(f"expected PIL.Image, got {type(image)}")
        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError(f"image has zero dimension: {image.size}")

        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            features = self.model(**inputs).image_embeds

        emb = features[0].float().cpu().numpy().astype(np.float32)
        return self._normalize(emb)

    def encode_joint(
        self, image: Image.Image, question: str, text_weight: float = 1.0
    ) -> np.ndarray:
        """
        Joint (image, question) embedding.

        Concatenates unit-norm CLIP image emb with unit-norm sentence emb, then
        re-normalizes. Without weighting, the image component (768-dim) contributes
        ~66.7% of raw magnitude vs ~33.3% for text (384-dim), so cosine similarity
        in FAISS is biased toward image similarity over question similarity.

        text_weight scales the sentence emb before concat to counteract this.
        text_weight=2.0 equalizes the per-component magnitude contribution
        (768 * 1.0 vs 384 * 2.0). Default is 1.0 (original behavior, image-dominant).
        Ablate this if cache precision on same-image/different-question pairs is poor.
        """
        img_emb = self.encode(image)
        sent_emb = get_sentence_encoder().encode(
            question,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        ).astype(np.float32)

        joint = np.concatenate([img_emb, sent_emb * text_weight])
        return self._normalize(joint)

    def encode_batch(self, images: list[Image.Image], chunk_size: int = 32) -> np.ndarray:
        """Image-only batch encoding. Returns (N, image_dim) normalized array."""
        if not images:
            return np.empty((0, self.image_dim), dtype=np.float32)

        results = []
        for i in range(0, len(images), chunk_size):
            chunk = [img.convert("RGB") for img in images[i : i + chunk_size]]
            inputs = self.processor(images=chunk, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                features = self.model(**inputs).image_embeds

            embs = features.float().cpu().numpy().astype(np.float32)
            results.append(self._normalize_batch(embs))

        return np.concatenate(results, axis=0)

    def encode_joint_batch(
        self, images: list[Image.Image], questions: list[str], text_weight: float = 1.0
    ) -> np.ndarray:
        """
        Joint batch encoding for (image, question) pairs.
        Returns (N, joint_dim) normalized array.
        See encode_joint for text_weight documentation.
        """
        if len(images) != len(questions):
            raise ValueError(
                f"images and questions must be same length, got {len(images)} and {len(questions)}"
            )
        if not images:
            return np.empty((0, self.joint_dim), dtype=np.float32)

        img_embs = self.encode_batch(images)
        sent_embs = get_sentence_encoder().encode(
            questions,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        ).astype(np.float32)

        joints = np.concatenate([img_embs, sent_embs * text_weight], axis=1)
        return self._normalize_batch(joints)
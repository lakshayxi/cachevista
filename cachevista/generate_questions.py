"""
Generates VQA answers for each COCO image using BLIP-VQA.
Saves to data/drift/questions.json.

Five fixed question types per image:
  describe  — "what is in this image?"
  count     — "how many objects are in this image?"
  color     — "what is the dominant color in this image?"
  spatial   — "what is in the background of this image?"
  yn        — "is this image taken outdoors?"

Run once before generate_drift_data.py.

Output format:
  {
    "question_templates": { "describe": "...", ... },   # stored once
    "images": {
      "000000001.jpg": {
        "answers": { "describe": "a cat on a mat", ... }
      },
      ...
    }
  }

NOTE: the answers field is not currently consumed by generate_drift_data.py,
which uses only the question strings (from QUESTION_TEMPLATES above) and
constructs drift labels from question type identity alone. The answers were
originally intended to ground labels in VLM output equivalence — same answer
for two queries → drift=0, different answer → drift=1. That design was not
implemented. The answers are stored here for future use or manual analysis.

If answer-based labelling is restored, generate_drift_data.py should compare
qdata["images"][img_name]["answers"][qt1] vs [qt2] instead of using question
type identity as the label signal. See generate_captions.py for context.

Fixed question templates mean all images share identical question strings.
This constrains hard negatives in generate_drift_data.py to differ only in
the sentence embeddings of these 5 fixed templates — image-adaptive questions
would produce richer hard negative boundaries but are not implemented.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

from cachevista.config import load
from cachevista.utils import get_device


QUESTION_TEMPLATES = {
    "describe": "what is in this image?",
    "count":    "how many objects are in this image?",
    "color":    "what is the dominant color in this image?",
    "spatial":  "what is in the background of this image?",
    "yn":       "is this image taken outdoors?",
}

_QUESTIONS = list(QUESTION_TEMPLATES.values())
_QTYPES = list(QUESTION_TEMPLATES.keys())


def generate_questions(img_dir: Path, out_path: Path, batch_size: int = 16,
                       skip_if_exists: bool = True):
    paths = sorted(img_dir.glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"no .jpg images found in {img_dir}")

    # load existing results for resumption — this script can take 20+ min;
    # a crash should not lose all progress
    images_done: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        images_done = existing.get("images", {})
        already_done = sum(1 for p in paths if p.name in images_done)
        if skip_if_exists and already_done == len(paths):
            print(f"questions already complete at {out_path}, skipping")
            return
        print(f"resuming — {already_done}/{len(paths)} already processed")

    device = get_device()
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    print(f"loading BLIP-VQA on {device} ({dtype})...")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base", torch_dtype=dtype
    )
    model = model.to(device)
    model.eval()
    print(f"BLIP-VQA loaded — processing {len(paths)} images...")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    pending = [p for p in paths if p.name not in images_done]

    for batch_start in range(0, len(pending), batch_size):
        batch_paths = pending[batch_start:batch_start + batch_size]

        # for each image, run all 5 questions in one batched forward pass
        # batch = (batch_size * n_questions) samples
        batch_images = []
        batch_questions = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            for q in _QUESTIONS:
                batch_images.append(img)
                batch_questions.append(q)

        inputs = processor(
            images=batch_images, text=batch_questions,
            return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            # num_beams=1 (greedy) — one output sequence per input guaranteed
            out = model.generate(**inputs, max_new_tokens=20, num_beams=1)

        raw_answers = processor.batch_decode(out, skip_special_tokens=True)

        n_q = len(_QUESTIONS)
        if len(raw_answers) != len(batch_paths) * n_q:
            raise RuntimeError(
                f"expected {len(batch_paths) * n_q} answers, "
                f"got {len(raw_answers)} — count mismatch"
            )

        for img_idx, p in enumerate(batch_paths):
            answers = {}
            for q_idx, qtype in enumerate(_QTYPES):
                answer = raw_answers[img_idx * n_q + q_idx].strip()
                if not answer:
                    # empty answer from BLIP — store placeholder so downstream
                    # code can detect and handle rather than silently using ""
                    answer = "__empty__"
                answers[qtype] = answer
            images_done[p.name] = {"answers": answers}

        # checkpoint every 10 batches
        if (batch_start // batch_size + 1) % 10 == 0:
            _write(out_path, images_done)
            done_count = batch_start + len(batch_paths)
            print(f"  {done_count}/{len(pending)} done (checkpointed)")

    _write(out_path, images_done)
    print(f"saved {len(images_done)} entries to {out_path}")

    # sanity check: show answers for the first image
    first_name = list(images_done.keys())[0]
    first = images_done[first_name]
    print(f"sample ({first_name}):")
    for qtype in _QTYPES:
        q = QUESTION_TEMPLATES[qtype]
        a = first["answers"][qtype]
        print(f"  [{qtype}] Q: {q}  A: {a}")


def _write(out_path: Path, images_done: dict):
    # question_templates stored once at top level, not repeated per image
    payload = {
        "question_templates": QUESTION_TEMPLATES,
        "images": images_done,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    cfg = load()
    img_dir = Path(cfg["data"]["coco_dir"])
    out_path = Path(cfg["data"]["drift_data_dir"]) / "questions.json"
    generate_questions(img_dir, out_path)
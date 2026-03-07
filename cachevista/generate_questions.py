"""
Generates questions and VQA answers for each COCO image using BLIP-VQA.
Saves to data/drift/questions.json.

Five question types per image:
  describe  — "what is in this image?"
  count     — "how many objects are in this image?"
  color     — "what is the dominant color in this image?"
  spatial   — "what is in the background of this image?"
  yn        — "is this image taken outdoors?"

Run once before generate_drift_data.py.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

from cachevista.config import load


QUESTION_TEMPLATES = {
    "describe": "what is in this image?",
    "count":    "how many objects are in this image?",
    "color":    "what is the dominant color in this image?",
    "spatial":  "what is in the background of this image?",
    "yn":       "is this image taken outdoors?",
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_questions(img_dir: Path, out_path: Path):
    device = get_device()
    print(f"loading BLIP-VQA on {device}...")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    model = model.to(device)
    model.eval()
    print("BLIP-VQA loaded")

    paths = sorted(img_dir.glob("*.jpg"))
    print(f"generating questions for {len(paths)} images...")

    results = {}
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        answers = {}

        for qtype, question in QUESTION_TEMPLATES.items():
            inputs = processor(images=img, text=question, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=20)
            answers[qtype] = processor.decode(out[0], skip_special_tokens=True)

        results[p.name] = {
            "questions": QUESTION_TEMPLATES,
            "answers": answers,
        }

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(paths)} done")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"saved {len(results)} entries to {out_path}")
    # show sample
    first = list(results.values())[0]
    for qtype, q in first["questions"].items():
        print(f"  [{qtype}] Q: {q}  A: {first['answers'][qtype]}")


if __name__ == "__main__":
    cfg = load()
    img_dir = Path(cfg["data"]["coco_dir"])
    out_path = Path(cfg["data"]["drift_data_dir"]) / "questions.json"
    generate_questions(img_dir, out_path)
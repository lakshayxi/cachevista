import json
import os
import urllib.request
from pathlib import Path


def download_coco_subset(out_dir: Path, n=500):
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_path = out_dir / "instances_val2017.json"
    if not ann_path.exists():
        print("downloading COCO val2017 annotations...")
        urllib.request.urlretrieve(
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            out_dir / "annotations.zip"
        )
        import zipfile
        with zipfile.ZipFile(out_dir / "annotations.zip") as z:
            z.extract("annotations/instances_val2017.json", out_dir)
        (out_dir / "annotations/instances_val2017.json").rename(ann_path)
        print("annotations ready")

    with open(ann_path) as f:
        coco = json.load(f)

    images = coco["images"][:n]
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    print(f"downloading {n} images...")
    for i, img in enumerate(images):
        dest = img_dir / img["file_name"]
        if dest.exists():
            continue
        try:
            urllib.request.urlretrieve(img["coco_url"], dest)
        except Exception as e:
            print(f"  skipped {img['file_name']}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}")

    print(f"done — images at {img_dir}")
    return img_dir


if __name__ == "__main__":
    download_coco_subset(Path("data/coco"), n=500)

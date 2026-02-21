import os
import json
import random
from tqdm import tqdm
from PIL import Image
import numpy as np

# ===========================
# CONFIG
# ===========================

COCO_ROOT = "data/raw/coco2017"
ANNOTATION_FILE = os.path.join(COCO_ROOT, "annotations", "instances_train2017.json")
IMAGE_DIR = os.path.join(COCO_ROOT, "train2017")

OUTPUT_DIR = "data/processed/ml_dataset"

TARGET_IMAGE_SIZE = 224
MIN_AREA = 10000  # Filter small boxes

MAX_PER_CLASS = 7000  # Adjustable

COCO_TO_WAREHOUSE = {
    "bottle": "FRAGILE",
    "wine glass": "FRAGILE",
    "cup": "FRAGILE",
    "suitcase": "HEAVY",
    "backpack": "HEAVY",
    "fire hydrant": "HAZARDOUS",
    "stop sign": "HAZARDOUS",
    "chair": "STANDARD",
    "book": "STANDARD"
}


# ===========================
# UTILS
# ===========================

def create_output_dirs():
    for split in ["train", "val", "test"]:
        for label in set(COCO_TO_WAREHOUSE.values()):
            os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)


def split_dataset(images):
    random.shuffle(images)
    train_split = int(0.8 * len(images))
    val_split = int(0.9 * len(images))

    return {
        "train": images[:train_split],
        "val": images[train_split:val_split],
        "test": images[val_split:]
    }


# ===========================
# MAIN
# ===========================

def build_dataset():

    print("Loading COCO annotations...")
    with open(ANNOTATION_FILE, "r") as f:
        coco = json.load(f)

    categories = coco["categories"]
    annotations = coco["annotations"]
    images = coco["images"]

    # Map category id to name
    id_to_name = {cat["id"]: cat["name"] for cat in categories}

    # Build category filter
    selected_cat_ids = {
        cat_id for cat_id, name in id_to_name.items()
        if name in COCO_TO_WAREHOUSE
    }

    print("Filtering annotations...")
    filtered_annotations = [
        ann for ann in annotations
        if ann["category_id"] in selected_cat_ids
        and ann["area"] >= MIN_AREA
        and ann["iscrowd"] == 0
    ]

    print(f"Total filtered annotations: {len(filtered_annotations)}")

    # Organize by warehouse class
    class_data = {}

    for ann in tqdm(filtered_annotations):
        cat_name = id_to_name[ann["category_id"]]
        warehouse_label = COCO_TO_WAREHOUSE[cat_name]

        if warehouse_label not in class_data:
            class_data[warehouse_label] = []

        class_data[warehouse_label].append(ann)

    create_output_dirs()

    print("Cropping and saving images...")

    for warehouse_label, anns in class_data.items():

        anns = anns[:MAX_PER_CLASS]

        split = split_dataset(anns)

        for split_name, split_anns in split.items():
            for ann in tqdm(split_anns):

                img_info = next(img for img in images if img["id"] == ann["image_id"])
                img_path = os.path.join(IMAGE_DIR, img_info["file_name"])

                try:
                    image = Image.open(img_path).convert("RGB")
                except:
                    continue

                x, y, w, h = ann["bbox"]
                crop = image.crop((x, y, x + w, y + h))
                crop = crop.resize((TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE))

                save_path = os.path.join(
                    OUTPUT_DIR,
                    split_name,
                    warehouse_label,
                    f"{ann['image_id']}_{ann['id']}.jpg"
                )

                crop.save(save_path)

    print("Dataset building complete.")


if __name__ == "__main__":
    build_dataset()
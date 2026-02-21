import os

BASE_DIR = "data/processed/ml_dataset"

for split in ["train", "val", "test"]:
    print(f"\n--- {split.upper()} ---")
    split_path = os.path.join(BASE_DIR, split)
    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)
        if os.path.isdir(cls_path):
            count = len(os.listdir(cls_path))
            print(f"{cls}: {count} images")
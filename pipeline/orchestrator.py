import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from ml.model import get_model
from vision.detector import WarehouseObjectDetector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("ml/saved_models/best_model.pth")
IMAGE_SIZE = 224
CLASS_NAMES = ["FRAGILE", "HAZARDOUS", "HEAVY", "STANDARD"]
DEFAULT_TEST_IMAGE = Path("data/processed/ml_dataset/test/FRAGILE/201436_84179.jpg")

_model = None
_transform = None
_detector = None


def _load_components():
    global _model, _transform, _detector

    if _model is None:
        _model = get_model(num_classes=4)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        _model.to(DEVICE)
        _model.eval()

    if _transform is None:
        _transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    if _detector is None:
        _detector = WarehouseObjectDetector(debug=False)

    return _model, _transform, _detector


def classify_crop(crop_image, model, transform):
    tensor = transform(crop_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item()

    return predicted_class, confidence_score


def run_pipeline(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        return {"error": f"Image not found: {image_path}"}

    if not MODEL_PATH.exists():
        return {"error": f"Model checkpoint not found: {MODEL_PATH}"}

    model, transform, detector = _load_components()
    image = Image.open(image_path).convert("RGB")
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    detections = detector.detect(cv_image)

    if not detections:
        return {"error": "No objects detected"}

    first_obj = detections[0]
    x, y, w, h = first_obj["bbox"]

    crop = image.crop((x, y, x + w, y + h))

    predicted_class, confidence = classify_crop(crop, model, transform)

    from rag.retriever import query_rag
    query = f"What are the handling instructions for {predicted_class.lower()} items?"
    answer, context = query_rag(query, predicted_class=predicted_class)

    result = {
        "image": str(image_path),
        "bbox": first_obj["bbox"],
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "handling_instructions": answer,
        "sources": [c["source"] for c in context]
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end warehouse pipeline")
    parser.add_argument("--image", type=Path, default=DEFAULT_TEST_IMAGE, help="Path to input image")
    args = parser.parse_args()

    output = run_pipeline(args.image)
    if "error" in output:
        print(output["error"])
        raise SystemExit(1)

    print("\n=== FINAL PIPELINE OUTPUT ===")
    for key, value in output.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

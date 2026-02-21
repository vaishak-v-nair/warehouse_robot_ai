import torch
from PIL import Image
import torchvision.transforms as transforms
import os
from vision.detector import WarehouseObjectDetector
from ml.model import get_model
from rag.retriever import query_rag

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ml/saved_models/best_model.pth"
IMAGE_SIZE = 224

model = get_model(num_classes=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

CLASS_NAMES = ["FRAGILE", "HAZARDOUS", "HEAVY", "STANDARD"]

detector = WarehouseObjectDetector(debug=False)


def classify_crop(crop_image):
    tensor = transform(crop_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item()

    return predicted_class, confidence_score


def run_pipeline(image_path):

    image = Image.open(image_path).convert("RGB")

    detections = detector.detect(image)

    if not detections:
        return {"error": "No objects detected"}

    first_obj = detections[0]
    x, y, w, h = first_obj["bbox"]

    crop = image.crop((x, y, x + w, y + h))

    predicted_class, confidence = classify_crop(crop)

    query = f"What are the handling instructions for {predicted_class.lower()} items?"
    answer, context = query_rag(query, predicted_class=predicted_class)

    result = {
        "image": image_path,
        "bbox": first_obj["bbox"],
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "handling_instructions": answer,
        "sources": [c["source"] for c in context]
    }

    return result


if __name__ == "__main__":

    test_image = "E:\warehouse_robot_ai\data\processed\ml_dataset\test\FRAGILE\201436_84179.jpg"

    if not os.path.exists(test_image):
        print(f"Image not found: {test_image}")
        exit()

    output = run_pipeline(test_image)

    print("\n=== FINAL PIPELINE OUTPUT ===")
    for k, v in output.items():
        print(f"{k}: {v}")
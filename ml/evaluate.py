import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import get_model

DATA_DIR = "data/processed/ml_dataset"
MODEL_PATH = "ml/saved_models/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    train_loader, val_loader, test_loader, class_names = get_dataloaders(DATA_DIR)
    num_classes = len(class_names)

    model = get_model(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    main()
import os
import torch
import numpy as np
from torch import nn, optim
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

from dataset import get_dataloaders
from model import get_model

DATA_DIR = "data/processed/ml_dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 8
LR = 1e-4
SAVE_PATH = "ml/saved_models/best_model.pth"

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def main():

    set_seed()

    train_loader, val_loader, test_loader, class_names = get_dataloaders(DATA_DIR)
    num_classes = len(class_names)

    model = get_model(num_classes).to(DEVICE)

    # Compute class weights
    class_counts = [len(train_loader.dataset.targets)]
    targets = train_loader.dataset.targets
    weights = compute_class_weight("balanced", classes=np.unique(targets), y=targets)
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

    best_val_acc = 0

    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        print(f"Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print("Model saved.")

    print("Training complete.")

if __name__ == "__main__":
    main()
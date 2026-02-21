import torch.nn as nn
import torchvision.models as models

def get_model(num_classes):

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last residual block
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
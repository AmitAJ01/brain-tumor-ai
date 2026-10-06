"""
EfficientNet-B0 Classification Model Architecture
Customized for 3-class brain tumor classification:
0 = glioma, 1 = meningioma, 2 = pituitary tumor
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from typing import Optional

CLASS_NAMES = ["glioma", "meningioma", "pituitary"]
CLASS_TO_IDX = {"glioma": 0, "meningioma": 1, "pituitary": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

def build_efficientnet_b0(num_classes: int = 3, pretrained: bool = True, dropout_rate: float = 0.3) -> nn.Module:
    """
    Construct EfficientNet-B0 classification network.
    Uses ImageNet pretrained backbone when available, replaces head with custom classifier.
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    # In EfficientNet-B0, features output has 1280 channels
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features, num_classes)
    )

    return model

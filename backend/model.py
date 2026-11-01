"""
model.py  -  The 1D CNN used for every ornament (Report, Section 4.2.5, Table 4.1)

    Input (3 x 100)  ->  Conv1D 3->16, k=7 + ReLU  ->  MaxPool(2)
                     ->  Conv1D 16->32, k=7 + ReLU ->  MaxPool(2)
                     ->  Conv1D 32->32, k=5 + ReLU ->  AdaptiveAvgPool1d(1)
                     ->  Flatten (32)  ->  Linear 32->1  ->  Sigmoid

The layer names ("features" and "classifier") must stay exactly like this,
because the saved .pth file stores the weights under these names.
Trainable parameters: 9,153 (same as the report).
"""

import json
import torch
import torch.nn as nn


class OrnamentCNN(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, arch["conv1_out_channels"], kernel_size=arch["conv1_kernel_size"]),   # features.0
            nn.ReLU(),                                                                          # features.1
            nn.MaxPool1d(2),                                                                    # features.2
            nn.Conv1d(arch["conv1_out_channels"], arch["conv2_out_channels"],
                      kernel_size=arch["conv2_kernel_size"]),                                   # features.3
            nn.ReLU(),                                                                          # features.4
            nn.MaxPool1d(2),                                                                    # features.5
            nn.Conv1d(arch["conv2_out_channels"], arch["conv3_out_channels"],
                      kernel_size=arch["conv3_kernel_size"]),                                   # features.6
            nn.ReLU(),                                                                          # features.7
            nn.AdaptiveAvgPool1d(1),                                                            # features.8
        )
        self.classifier = nn.Linear(arch["conv3_out_channels"], arch["classifier_output"])

    def forward(self, x):
        # x shape: (number_of_frames, 3, 100)
        x = self.features(x)          # -> (N, 32, 1)
        x = torch.flatten(x, 1)       # -> (N, 32)
        return self.classifier(x)     # -> (N, 1)  logits (sigmoid is applied later)


def load_ornament_model(config_path, model_path):
    """Read the JSON config, build the CNN and load the trained weights."""
    with open(config_path, "r") as f:
        config = json.load(f)

    model = OrnamentCNN(config["architecture"])
    weights = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model.eval()   # evaluation mode: no training behaviour
    return model, config

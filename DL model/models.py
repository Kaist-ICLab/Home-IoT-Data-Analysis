# DL model python implementation
# models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# CNN1d

def make_cnn_layers():
        layers = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(7)
        )
        return layers


class CNN1d(nn.Module):
    def __init__(self):
        super(CNN1d, self).__init__()

        self.encoder = make_cnn_layers()

        self.classifier = nn.Sequential(
            nn.Linear(448, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)

        x = x.view(x.size(0), -1)  # Reshape to flatten
        x = self.classifier(x)
        return x


class CNN1dAttn(nn.Module):
    def __init__(self, num_heads=4, attention_dim=64):
        super(CNN1dAttn, self).__init__()
        self.encoder = make_cnn_layers()
        
        self.attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, batch_first=True)
        # Classifier for final prediction
        self.classifier = nn.Sequential(
            nn.Linear(448, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.encoder(x)
        
        # Reshape for attention layer: (batch_size, seq_length, feature_dim)
        x = x.permute(0, 2, 1)
        
        attn_output, _ = self.attention(x, x, x)

        # Flatten the attention output for the classifier
        attn_output = attn_output.permute(0, 2, 1).contiguous()
        attn_output = attn_output.view(attn_output.size(0), -1)
        
        # Classification
        x = self.classifier(attn_output)
        return x


class FusionBase(nn.Module):
    def __init__(self, models, input_slices, ):
        super(FusionBase, self).__init__()
        self.models = nn.ModuleList(models)
        self.input_slices = input_slices

        # Remove last linear and sigmoid layer
        for model in self.models:
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])

        self.classifier = nn.Sequential(
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        outputs = []

        for model, slice_ in zip(self.models, self.input_slices):
            x_input = x[:, slice_]
            output = model(x_input)
            outputs.append(output)

        fused_output = torch.cat(outputs, dim=1)

        x = self.classifier(fused_output)
        return x
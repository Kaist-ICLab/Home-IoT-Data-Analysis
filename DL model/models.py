# DL model python implementation

# models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DNN_small(nn.Module):
    def __init__(self):
        super(DNN_small, self).__init__()
        self.classifier = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

class DNN_medium(nn.Module):
    def __init__(self):
        super(DNN_medium, self).__init__()
        self.classifier = nn.Sequential(
            nn.LazyBatchNorm1d(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

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


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        b, c, _ = x.size()
        y = x.mean(dim=2)  # Global average pooling
        y = self.fc1(y)
        y = torch.relu(y)
        y = self.fc2(y)
        y = torch.sigmoid(y).unsqueeze(2)
        return x * y

class CNN1d(nn.Module):
    def __init__(self, use_se_block=False):
        super(CNN1d, self).__init__()

        self.encoder = make_cnn_layers()

        self.use_se_block = use_se_block
        if self.use_se_block:
            self.se_block = SqueezeExcitation(64)

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

        if self.use_se_block:
            x = self.se_block(x)

        x = x.view(x.size(0), -1)  # Reshape to flatten
        x = self.classifier(x)
        return x

class CNN1dCat(nn.Module):
    def __init__(self):
        super(CNN1dCat, self).__init__()
        self.features_c = make_cnn_layers()
        self.features_ms = make_cnn_layers()
        self.features_mfcc = make_cnn_layers()

        self.classifier = nn.Sequential(
            nn.Linear(448 * 3, 64),  # Updated input size based on the concatenated features
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_c = x[:, [i for i in range(12)]].unsqueeze(1)
        x_ms = x[:, [i for i in range(12, 140)]].unsqueeze(1)
        x_mfcc = x[:, [i for i in range(140, 180)]].unsqueeze(1)
        x_c = self.features_c(x_c)
        x_ms = self.features_ms(x_ms)
        x_mfcc = self.features_mfcc(x_mfcc)

        x_c = x_c.view(x_c.size(0), -1)  # Reshape to flatten
        x_ms = x_ms.view(x_ms.size(0), -1)
        x_mfcc = x_mfcc.view(x_mfcc.size(0), -1)
        
        x = torch.cat((x_c, x_ms, x_mfcc), dim=1)  # Concatenate features along the channel dimension
        x = self.classifier(x)
        return x

class TemporalSelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(TemporalSelfAttention, self).__init__()
        self.conv1d = nn.Conv1d(input_dim, attention_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, feature_dim)
        
        # Project the input using a convolutional layer to get attention scores
        attn_scores = self.conv1d(x.permute(0, 2, 1))  # Shape: (batch_size, attention_dim, seq_length)
        attn_scores = attn_scores.permute(0, 2, 1)  # Shape: (batch_size, seq_length, attention_dim)
        
        # Softmax over the sequence dimension to get attention weights
        attn_weights = self.softmax(attn_scores)  # Shape: (batch_size, seq_length, attention_dim)

        # Apply the attention weights to the input sequence
        attn_output = x * attn_weights  # Element-wise multiplication (batch_size, seq_length, feature_dim)
        
        return attn_output


class CNN1dAttn(nn.Module):
    def __init__(self, num_heads=4, attention_dim=64, 
                 attention_type="multihead",  # "multihead" 또는 "temporal"
                 attention_mechanism="none"): # "feature-wise", "cross-attention", "transformer", "none"
        super(CNN1dAttn, self).__init__()
        self.encoder = make_cnn_layers()
        
        # Attention 종류 선택
        self.attention_type = attention_type
        if attention_type == "multihead":
            self.attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, batch_first=True)
        elif attention_type == "temporal":
            self.attention = TemporalSelfAttention(input_dim=64, attention_dim=attention_dim)
        
        # 추가 Attention 메커니즘 선택
        self.attention_mechanism = attention_mechanism
        if attention_mechanism == "feature-wise":
            self.feature_attention = nn.Sequential(
                nn.Linear(attention_dim, attention_dim),  # `feature_dim`에 맞게 설정
                nn.Sigmoid()
            )
        elif attention_mechanism == "cross-attention":
            self.cross_attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, batch_first=True)
        elif attention_mechanism == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=attention_dim, nhead=num_heads, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

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
        
        # Attention 적용
        if self.attention_type == "multihead":
            attn_output, _ = self.attention(x, x, x)
        elif self.attention_type == "temporal":
            attn_output = self.attention(x)
        
        # 추가 메커니즘 적용: feature-wise, cross-attention, transformer
        if self.attention_mechanism == "feature-wise":
            # Feature-wise attention 적용
            feature_weights = self.feature_attention(attn_output.mean(dim=1))  # `mean`으로 seq_length 축소
            attn_output = attn_output * feature_weights.unsqueeze(1)
        elif self.attention_mechanism == "cross-attention":
            cross_output, _ = self.cross_attention(attn_output, attn_output, attn_output)
            attn_output = attn_output + cross_output  # Skip connection
        elif self.attention_mechanism == "transformer":
            attn_output = self.transformer_encoder(attn_output)

        # Flatten the attention output for the classifier
        attn_output = attn_output.permute(0, 2, 1).contiguous()
        attn_output = attn_output.view(attn_output.size(0), -1)
        
        # Classification
        x = self.classifier(attn_output)
        return x


# class CNN1dAttn(nn.Module):
#     def __init__(self, num_heads=4, attention_dim=64):
#         super(CNN1dAttn, self).__init__()
#         self.encoder = make_cnn_layers()
        
#         self.attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, batch_first=True)
        
#         self.classifier = nn.Sequential(
#             nn.Linear(448, 64),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(32, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.encoder(x)
        
#         # Prepare for attention layer
#         x = x.permute(0, 2, 1)  # Change shape to (batch_size, seq_length, feature_dim)
        
#         # Apply attention
#         attn_output, _ = self.attention(x, x, x)
        
#         # Flatten the attention output
#         attn_output = attn_output.permute(0, 2, 1).contiguous()
#         attn_output = attn_output.view(attn_output.size(0), -1)
        
#         # Classifier
#         x = self.classifier(attn_output)
#         return x

# class FusionBase(nn.Module):
#     def __init__(self, models, input_slices):
#         super(FusionBase, self).__init__()
#         self.models = nn.ModuleList(models)
#         self.input_slices = input_slices

#         # Removing the last Linear and Sigmoid layers from each model
#         for model in self.models:
#             model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])

#         self.classifier = nn.Sequential(
#             nn.LazyLinear(32),
#             nn.ReLU(),
#             nn.BatchNorm1d(32),
#             nn.Linear(32, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         outputs = []
#         for model, slice_ in zip(self.models, self.input_slices):
#             x_input = x[:, slice_]
#             output = model(x_input)
#             outputs.append(output)
        
#         x = torch.cat(outputs, dim=1)
#         x = self.classifier(x)
#         return x

class FusionBase(nn.Module):
    def __init__(self, models, input_slices, 
                 classifier_option="original",  # "original", "complex", "simple" 중 하나 선택
                 use_attention_weights=False):  # Attention 가중치 사용 여부
        super(FusionBase, self).__init__()
        self.models = nn.ModuleList(models)
        self.input_slices = input_slices
        self.use_attention_weights = use_attention_weights

        # 각 모델의 classifier에서 마지막 Linear와 Sigmoid를 제거
        for model in self.models:
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])

        # classifier_option에 따라 다른 classifier 설정
        if classifier_option == "original":
            self.classifier = nn.Sequential(
                nn.LazyLinear(32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        elif classifier_option == "complex":
            self.classifier = nn.Sequential(
                nn.LazyLinear(64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(p=0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(p=0.3),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        elif classifier_option == "simple":
            self.classifier = nn.Sequential(
                nn.LazyLinear(16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

        else:
            raise ValueError("Invalid classifier_option. Choose from 'original', 'complex', 'simple'.")

        # Attention 가중치 초기화
        if self.use_attention_weights:
            self.attn_weights = nn.Parameter(torch.ones(len(models), requires_grad=True))

    def forward(self, x):
        outputs = []

        # 각 모델에 대해 입력 슬라이싱 및 전방향 계산
        for model, slice_ in zip(self.models, self.input_slices):
            x_input = x[:, slice_]
            output = model(x_input)
            outputs.append(output)

        # Attention 가중치 적용 여부
        if self.use_attention_weights:
            attn_weights = torch.softmax(self.attn_weights, dim=0)
            fused_output = torch.cat([attn_weights[i] * outputs[i] for i in range(len(outputs))], dim=1)
        else:
            fused_output = torch.cat(outputs, dim=1)

        # 최종 classifier를 통해 결과 계산
        x = self.classifier(fused_output)
        return x
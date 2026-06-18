"""
models.py
=========
Архитектуры нейронных сетей: CNN Baseline, Transfer Learning (ResNet18) и DenseNet.

Содержит три класса моделей для классификации спектрограмм:
    - CNNBaseline: базовая свёрточная сеть с нуля
    - TransferLearningModel: ResNet18 с предобучением на ImageNet
    - DenseNetModel: DenseNet121 с предобучением на ImageNet
"""

import torch
import torch.nn as nn
import torchvision.models as models
from config import config

class CNNBaseline(nn.Module):
    """
    Базовая свёрточная нейросеть для классификации спектрограмм.
    
    Архитектура:
        - 4 свёрточных блока (Conv2d + BatchNorm + ReLU + MaxPool2d)
        - AdaptiveAvgPool2d для приведения к фиксированному размеру (4x4)
        - 2 полносвязных слоя с Dropout
    
    Вход: (batch, 1, 128, time_frames)
    Выход: (batch, 2) — логиты для классов [healthy, parkinson]
    
    Количество параметров: ~2.5M
    """
    def __init__(self, num_classes=2):
        super(CNNBaseline, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT), nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Dropout(config.DROPOUT), nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class TransferLearningModel(nn.Module):
    """
    Модель с Transfer Learning на базе ResNet18.
    
    Загружает предобученные на ImageNet веса, адаптирует первый слой под 1 канал,
    заменяет классификатор для 2 классов, замораживает слои до layer4.
    
    Вход: (batch, 1, 128, time_frames) → интерполируется до (3, 224, 224)
    Выход: (batch, 2)
    
    Количество параметров: ~27M
    """
    def __init__(self, num_classes=2):
        super(TransferLearningModel, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(config.DROPOUT), nn.Linear(num_features, 256), nn.ReLU(),
            nn.Dropout(config.DROPOUT), nn.Linear(256, num_classes)
        )
        for name, param in self.backbone.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)


class DenseNetModel(nn.Module):
    """
    Модель с Transfer Learning на базе DenseNet121.
    
    Особенности:
        - Загружает предобученные на ImageNet веса
        - Адаптирует первый слой под 1 канал (спектрограмма)
        - Заменяет классификатор для 2 классов
        - Замораживает слои до denseblock4 (тренируется только denseblock4 и classifier)
    
    Вход: (batch, 1, 128, time_frames)
    Выход: (batch, 2)
    """
    def __init__(self, num_classes=2):
        super(DenseNetModel, self).__init__()
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.backbone.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(256, num_classes)
        )
        for name, param in self.backbone.named_parameters():
            if 'denseblock4' not in name and 'classifier' not in name:
                param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)


def get_model(model_name, num_classes=2):
    """
    Фабрика моделей — возвращает модель по имени.
    
    Параметры:
        model_name (str): Название модели ("cnn_baseline", "transfer_learning", "densenet").
        num_classes (int, optional): Количество классов. По умолчанию 2.
    
    Возвращает:
        nn.Module: Экземпляр модели.
    
    Исключения:
        ValueError: Если название модели неизвестно.
    """
    if model_name == "cnn_baseline":
        return CNNBaseline(num_classes)
    elif model_name == "transfer_learning":
        return TransferLearningModel(num_classes)
    elif model_name == "densenet":
        return DenseNetModel(num_classes)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")
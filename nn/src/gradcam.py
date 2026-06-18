"""
gradcam.py
==========
Grad-CAM визуализация внимания модели на спектрограмме.

Поддерживает CNN Baseline, Transfer Learning (ResNet18) и DenseNet.
Позволяет увидеть, на какие области спектрограммы модель обращает внимание.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy.ndimage import zoom


class GradCAM:
    """
    Grad-CAM для визуализации важных областей спектрограммы.
    
    Использует градиенты последнего свёрточного слоя для построения
    тепловой карты внимания.
    """    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_backward_hook(self._save_gradients)
    
    def _save_activations(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        """
        Генерирует Grad-CAM тепловую карту.
        
        Параметры:
            input_tensor (torch.Tensor): Входная спектрограмма.
            target_class (int, optional): Целевой класс. По умолчанию None (берётся предсказанный).
        
        Возвращает:
            np.ndarray: Тепловая карта внимания.
        """
        self.model.eval()
        
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(1).item()
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.nn.functional.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam.cpu().numpy()


def get_target_layer(model, model_name):
    """
    Возвращает последний свёрточный слой для Grad-CAM.
    
    Параметры:
        model (nn.Module): Модель (CNNBaseline, TransferLearningModel или DenseNetModel).
        model_name (str): "cnn_baseline", "transfer_learning" или "densenet".
    
    Возвращает:
        nn.Module: Последний свёрточный слой.
    
    Исключения:
        ValueError: Если название модели неизвестно.
    """
    if model_name == "cnn_baseline":
        return model.features[-3]  # Последний Conv2d перед AdaptiveAvgPool2d
    
    elif model_name == "transfer_learning":
        return model.backbone.layer4[-1].conv2  # Последний свёрточный слой ResNet18
    
    elif model_name == "densenet":
        return model.backbone.features.denseblock4.denselayer16  # Последний сверточный слой DenseNet
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def visualize_gradcam(model, model_name, spectrogram, label, save_path, class_names=['Healthy', 'Parkinson']):
    """
    Визуализирует Grad-CAM поверх спектрограммы.
    
    Параметры:
        model (nn.Module): Обученная модель.
        model_name (str): "cnn_baseline", "transfer_learning" или "densenet".
        spectrogram (np.ndarray): Спектрограмма (128, time).
        label (int): Истинная метка (0 - healthy, 1 - parkinson).
        save_path (Path): Путь для сохранения графика.
        class_names (list, optional): Названия классов. По умолчанию ['Healthy', 'Parkinson'].
    """
    model.eval()
    
    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)
    
    input_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(1).item()
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
    
    heatmap = gradcam.generate(input_tensor, target_class=pred)
    
    heatmap_resized = zoom(heatmap, 
                          (spectrogram.shape[0] / heatmap.shape[0],
                           spectrogram.shape[1] / heatmap.shape[1]))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Исходная спектрограмма
    axes[0].imshow(spectrogram, aspect='auto', origin='lower')
    axes[0].set_title(f'Спектрограмма\nИстинная метка: {class_names[label]}')
    axes[0].set_xlabel('Временные фреймы')
    axes[0].set_ylabel('Mel-частоты')
    
    # 2. Тепловая карта внимания
    im = axes[1].imshow(heatmap_resized, cmap='hot', aspect='auto', origin='lower')
    axes[1].set_title('Внимание модели (Grad-CAM)')
    axes[1].set_xlabel('Временные фреймы')
    axes[1].set_ylabel('Mel-частоты')
    plt.colorbar(im, ax=axes[1], label='Важность')
    
    # 3. Наложение
    axes[2].imshow(spectrogram, cmap='gray', aspect='auto', origin='lower', alpha=0.6)
    axes[2].imshow(heatmap_resized, cmap='hot', aspect='auto', origin='lower', alpha=0.4)
    axes[2].set_title(f'Наложение внимания\nПредсказание: {class_names[pred]} ({probs[pred]:.2f})')
    axes[2].set_xlabel('Временные фреймы')
    axes[2].set_ylabel('Mel-частоты')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
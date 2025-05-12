# 用于封装模型加载与掩码预测
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

from models.u2net import U2NET  # 模型结构

class U2NetPredictor:
    def __init__(self, model_path, device='cpu'):
        self.model = U2NET(3, 1)
        self.device = torch.device(device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def predict_mask(self, image_path):
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            d1, _, _, _, _, _, _ = self.model(img_tensor)
            pred = d1[:, 0, :, :]
            pred = F.upsample(pred.unsqueeze(0), size=image.size[::-1], mode='bilinear').squeeze().cpu().numpy()

        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        mask = (pred * 255).astype(np.uint8)
        mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]
        return mask

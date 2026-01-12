import os
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from model_service.pytorch_model_service import PTServingBaseService

# ================== 定义LeNet5 ==================
class LeNet5(nn.Module):
    def __init__(self, num_classes=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ================== 图像预处理 ==================
infer_transformation = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ================== 加载模型 ==================
def load_model(model_path):
    model = LeNet5()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# ================== 推理服务 ==================
class PTVisionService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        super(PTVisionService, self).__init__(model_name, model_path)
        self.model, self.device = load_model(model_path)
        self.labels = [0,1,2,3,4,5,6,7,8,9]

    # 数据预处理
    def _preprocess(self, data):
        preprocessed_data = {}
        for key, val in data.items():
            batch = []
            for file_name, file_content in val.items():
                img = Image.open(file_content).convert("L")
                tensor = infer_transformation(img).unsqueeze(0)  # 添加 batch 维度
                batch.append(tensor)
            batch_tensor = torch.cat(batch, dim=0).to(self.device)
            preprocessed_data[key] = batch_tensor
        return preprocessed_data

    # 模型推理
    def _inference(self, data):
        result = {}
        with torch.no_grad():
            for key, batch in data.items():
                output = self.model(batch)
                result[key] = output
        return result

    # 后处理
    def _postprocess(self, data):
        results = []
        for key, output in data.items():
            pred = torch.argmax(output, dim=1)
            result = {key: [self.labels[i] for i in pred.cpu().numpy()]}
            results.append(result)
        return results

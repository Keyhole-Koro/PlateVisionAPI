from services.recognition.ocr_interface import OCREngine, MODEL_DIR

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

CHAR_CNN_DIR = os.path.join(MODEL_DIR, "char_cnn")


class CharCNN(nn.Module):
    def __init__(self, num_classes=46):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 64x64 -> 32x32
        x = self.pool(F.relu(self.conv2(x)))  # 32x32 -> 16x16
        x = self.dropout(x)
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CharCNNEngine(OCREngine):
    hiragana_chars = [
        'あ', 'い', 'う', 'え', 'お',
        'か', 'き', 'く', 'け', 'こ',
        'さ', 'し', 'す', 'せ', 'そ',
        'た', 'ち', 'つ', 'て', 'と',
        'な', 'に', 'ぬ', 'ね', 'の',
        'は', 'ひ', 'ふ', 'へ', 'ほ',
        'ま', 'み', 'む', 'め', 'も',
        'や',       'ゆ',       'よ',
        'ら', 'り', 'る', 'れ', 'ろ',
        'わ',                   'を',
                  'ん'
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert NumPy array to tensor
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64))
    ])
    
    def __init__(self, model_name="hiragana"):
        model = CharCNN(num_classes=len(self.hiragana_chars))
        model_path = os.path.join(CHAR_CNN_DIR, model_name + ".pth")
        # Load the model on the CPU
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model = model

    async def recognize_text(self, image):
        # Use the transform pipeline directly on the NumPy array
        tensor_img = self.transform(image).unsqueeze(0)  # Add batch dimension
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor_img)
            pred = torch.argmax(output, dim=1).item()
            return self.hiragana_chars[pred]
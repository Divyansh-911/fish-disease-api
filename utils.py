import torch
from torchvision import models, transforms
from PIL import Image

def load_model():
    model = models.resnet50(pretrained=False)
    num_classes = 12
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("model/fish_model.pt", map_location="cpu"))
    model.eval()
    with open("model/labels.txt") as f:
        labels = [line.strip() for line in f]
    return model, labels

def predict(image: Image.Image, model, labels):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        return labels[predicted.item()]

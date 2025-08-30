import torch
from torchvision import transforms
from PIL import Image
import json

def load_cv_model(model_path: str, class_to_idx_path: str):
    from torchvision import models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    with open(class_to_idx_path, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    return model, idx_to_class, device

def predict_image_prob(model, device, image_path: str):
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    # assuming classes: ['diseased','healthy'] or vice versa - we will map dynamically outside
    return probs

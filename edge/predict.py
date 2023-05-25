import torch
from torchvision import models
from torchvision.transforms import ToTensor
from PIL import Image
import sys

# We assume that the first argument is the image file and the second is the model file
image_path = sys.argv[1]
model_path = sys.argv[2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = torch.load(model_path)
model = model.to(device)
model.eval()

# Open and preprocess image
image = Image.open(image_path)
transform = ToTensor()
image = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(image)
    prob = torch.sigmoid(output).item()

print(prob)

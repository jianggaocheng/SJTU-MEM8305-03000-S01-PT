import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.nn import Linear, Hardswish, Dropout
from PIL import Image
import sys

# Get a set of pretrained model weights
weights = models.MobileNet_V3_Small_Weights.DEFAULT

# Get the data transformations used to create our pretrained weights
auto_transforms = weights.transforms()

def predict_image(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model structure
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    # Open image
    image = Image.open(image_path)

    # Convert images to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply the transformations
    image = auto_transforms(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = torch.sigmoid(model(image)).item()
    
    return output

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python predict.py image_path model_path")
        sys.exit()

    image_path = sys.argv[1]
    model_path = sys.argv[2]

    print(predict_image(image_path, model_path))

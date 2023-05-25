import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.nn import Linear, Hardswish, Dropout
from PIL import Image
import os
import glob

# Get a set of pretrained model weights
weights = models.MobileNet_V3_Small_Weights.DEFAULT

# Get the data transformations used to create our pretrained weights
auto_transforms = weights.transforms()

def predict_image(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model structure
    model = models.mobilenet_v3_small()

    # Update the classifier to match the one you used during training
    model.classifier = torch.nn.Sequential(
        Linear(in_features=576, out_features=1024, bias=True),
        Hardswish(),
        Dropout(p=0.2, inplace=True), 
        Linear(in_features=1024, out_features=1, bias=True)
    )

    # Load the state dict into the model structure
    model.load_state_dict(torch.load(model_path))
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
    output = model(image)
    # Apply sigmoid and round to transform the output to either 0 or 1
    prediction = torch.round(torch.sigmoid(output))

    # Note that prediction can be either 0 or 1 since it is a binary classification problem
    preds = torch.where(output > 0.5, torch.tensor([0]).to(device), torch.tensor([1]).to(device))

    if preds[0] == 1:
        print('Fire detected in image')
    else:
        print('No fire detected in image')

def predict_images_in_directory(directory, model_path):
    # Get a list of all .png files in the directory
    image_files = glob.glob(os.path.join(directory, '*.*'))

    # Loop over all image files
    for image_path in image_files:
        print(f"Path: {image_path}")
        predict_image(image_path, model_path)

if __name__ == '__main__':
    # image_path = sys.argv[1]
    image_path = './fire.13.png'
    model_path = './best_model_tl.pth'  # update this to the path of your model
    
    predict_images_in_directory("./Fire-Detection/0", model_path)
    predict_images_in_directory("./Fire-Detection/1", model_path)

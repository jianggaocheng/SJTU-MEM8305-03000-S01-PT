# Import modules
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split 
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torchinfo import summary

import os
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# For macOS, delete .DS_Store file to prevent error
os.system('find . -name .DS_Store | xargs rm -rf')

print(f"PyTorchVersion: {torch.__version__}")

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Setup path for our data files
# Define data root directory
root_dir = Path("./data")
fire_dir = Path("./data/fire_images")
non_fire_dir = Path("./data/non_fire_images")

# You can also apply data transformation using the `transforms` parameter here
fire_dataset = ImageFolder(root_dir)

# Split the dataset into training, validation and testing
train_dataset, valid_dataset, test_dataset = random_split(fire_dataset, [0.6, 0.2, 0.2], generator=torch.Generator())
print(f"train_dataset: {len(train_dataset)}, valid_dataset: {len(valid_dataset)}, test_dataset: {len(test_dataset)}")

# Define custom data subset
class MySubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        
        # If a transform argument is provided, apply the transformaiton to the subset
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)
    
# Define function for training and evaluating each epoch
def train_per_epoch(model, train_dataloader, test_dataloader, loss_fn, optimizer):

    model.train()
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(train_dataloader):

        X, y = X.to(device), torch.unsqueeze(y, dim=1).float().to(device)

        # Make prediction
        y_logits = model(X)
        y_pred = 1 - torch.round(torch.sigmoid(y_logits))

        # Calculate and accumulate loss
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()

        # Calculate and accumulate accuracy
        acc = (y_pred == 1 - y).sum().item() / len(y)  # Reverse the label for accuracy calculation
        train_acc += acc

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate the average loss and average accuracy by batch (in one epoch)
    train_loss = train_loss / len(train_dataloader)
    train_acc = train_acc / len(train_dataloader)

    ### Testing Loop
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), torch.unsqueeze(y, dim=1).float().to(device)

            test_logits = model(X)
            test_pred = 1 - torch.round(torch.sigmoid(test_logits))

            loss = loss_fn(test_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            acc = (test_pred == 1 - y).sum().item() / len(y)  # Reverse the label for accuracy calculation
            test_acc += acc

    # Calculate the average loss and average accuracy by batch (in one epoch)
    test_loss = test_loss / len(test_dataloader)
    test_acc = test_acc / len(test_dataloader)
    return train_loss, train_acc, test_loss, test_acc

# Setup ModelCheckpoint
def checkpoint(model, filename):
    torch.save(model, filename)
    
def resume(filename):
    model = torch.load(filename)
    return model

# Define train function
def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, best_model_path):
    
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    
    # Setup early stopping and model checkpoint
    best_loss = float("inf")
    best_epoch = -1
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, test_loss, test_acc = train_per_epoch(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optimizer)
        
        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        # EarlyStopping and ModelCheckpoint
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            checkpoint(model, best_model_path)
    
    print(f"Epoch with best loss: Epoch {best_epoch+1} at {best_loss:.4f}")

    return results

# Get a set of pretrained model weights
weights = models.MobileNet_V3_Small_Weights.DEFAULT

# Get the data transformations used to create our pretrained weights
auto_transforms = weights.transforms()

# Apply data transformation only to the training and validation dataset
train_dataset = MySubset(train_dataset, transform=auto_transforms)
valid_dataset = MySubset(valid_dataset, transform=auto_transforms)

# Load the weights into the model
model_tl = models.mobilenet_v3_small(weights=weights).to(device)
summary(model=model_tl, input_size=(32, 3, 384, 384))

# Redefine classifier layer
model_tl.classifier = torch.nn.Sequential(
    nn.Linear(in_features=576, out_features=1024, bias=True),
    nn.Hardswish(),
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1024, out_features=1, bias=True)).to(device)

summary(model=model_tl, input_size=(32, 3, 384, 384))

# Hyper-parameter adjustment and optimizer initialization
batch_size = 32
epochs_num = 30
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model_tl.parameters(), lr=0.0001)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

#train model_tl
model_tl_results = train(model=model_tl, train_dataloader=train_loader, test_dataloader=valid_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=epochs_num, best_model_path = "best_model_tl.pth")

def plot_results(results):
    
    train_loss = results["train_loss"]
    train_acc = results["train_acc"]
    test_loss = results["test_loss"]
    test_acc = results["test_acc"]
    
    epochs = range(len(results['train_loss']))
    
    plt.figure(figsize=(10, 5))
    
    #plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    #plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='train_accuracy')
    plt.plot(epochs, test_acc, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

    plt.show()

plot_results(model_tl_results)
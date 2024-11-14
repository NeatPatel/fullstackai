import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the neural network model (same structure as used during training)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

# Load the saved model
model = torch.load('mnist_full_model.pth', map_location=device)
model.to(device)
# Set model to evaluation mode
model.eval()

# Preprocess the input image
def preprocess_image(image_path):
    # series of transforms
    transform = transforms.Compose([
        transforms.Grayscale(),  # greyscale the image
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize for MNIST
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.view(-1, 28*28)  # Flatten the image to match input size (784)
    return image.to(device)

# predict digit
def predict(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)  # Get the class with the highest score
    return predicted.item()


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784  # 28 x 28
hidden_size = 100
num_classes = 10
batch_size = 64

# MNIST dataset
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# redefine NN class
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

# Evaluate the model on the test dataset
n_correct = 0
n_samples = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get class prediction
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()


accuracy = 100.0 * n_correct / n_samples
print(f'Accuracy of the model on the test dataset: {accuracy:.2f}%')

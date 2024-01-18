import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
from MobileNetV3 import MobileNetV3V  # make sure in the same folder
from tqdm import tqdm
import matplotlib.pyplot as plt

# define hyperpara
batch_size = 128
epochs = 300
learning_rate = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# preprocessing
NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
tr_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*NORM)
])

te_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*NORM)
])

# load dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tr_transforms)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=te_transforms)

# split dataset
test_size = len(testset) // 2
valid_size = len(testset) - test_size
test_dataset, valid_dataset = random_split(testset, [test_size, valid_size], generator=torch.Generator().manual_seed(123))

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

# initial model
net = MobileNetV3V(num_classes=10).to(device)

# define model
criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=5e-4)
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)



# early-stop
early_stopping_patience = 60
min_val_loss = float('inf')
epochs_no_improve = 0

# train function
def train_epoch(loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data in tqdm(loader, desc='Training', leave=False):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    train_loss = running_loss / len(loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# test function
def test_epoch(loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(loader, desc='Testing', leave=False):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_loss = running_loss / len(loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc


# store the loss values
train_losses = []
valid_losses = []

# train model
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_loss, train_acc = train_epoch(trainloader, net, criterion, optimizer, device)
    valid_loss, valid_acc = test_epoch(validloader, net, criterion, device)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_acc:.2f}%")

    # store the loss values
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # early stop justification
    if valid_loss < min_val_loss:
        min_val_loss = valid_loss
        epochs_no_improve = 0
        torch.save(net.state_dict(), '/home/yushan/adaptive_personalization/motivation_exp/model/mobilenet_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == early_stopping_patience:
            print('Early stopping!')
            break
    scheduler.step()

print('Finished Training')

# load the best model
net.load_state_dict(torch.load('/home/yushan/adaptive_personalization/motivation_exp/model/mobilenet_model.pth'))

# evaluation
test_loss, test_acc = test_epoch(testloader, net, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

# figure
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/home/yushan/adaptive_personalization/motivation_exp/fig/loss_curve_mobilenet.png')
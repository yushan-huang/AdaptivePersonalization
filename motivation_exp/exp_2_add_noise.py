import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from ResNet import ResNetCifar  # make sure in the same folder
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import chain



# define initial hyperpara
batch_size = 128
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

# utilize 1/10 original trainset to mimic fewer local samples 
subset_size = len(trainset) // 10
indices = torch.randperm(len(trainset)).tolist()
subset_indices = indices[:subset_size]
subset_trainset = Subset(trainset, subset_indices)
subset_trainloader = DataLoader(subset_trainset, batch_size=batch_size, shuffle=True, num_workers=10)

# split valid and test dataset
test_size = len(testset) // 2
valid_size = len(testset) - test_size
test_dataset, valid_dataset = random_split(testset, [test_size, valid_size], generator=torch.Generator().manual_seed(123))

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)


net = ResNetCifar(depth=26).to(device)

# define model
criterion = nn.CrossEntropyLoss()

# load model
net.load_state_dict(torch.load('/home/yushan/adaptive_personalization/motivation_exp/model/resnet26_model.pth'))


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

def add_noise_to_layer(layer, std_dev=0.01):
    with torch.no_grad(): 
        for param in layer.parameters():
            noise = torch.randn_like(param) * std_dev 
            param.add_(noise)  


# test acc without noise
_, acc_without_noise = test_epoch(testloader, net, criterion, device)

# Add noise
# add_noise_to_layer(net.conv1, std_dev=0.15)
# add_noise_to_layer(net.layer1, std_dev=0.15)

add_noise_to_layer(net.fc, std_dev=6)

_, acc_with_noise = test_epoch(testloader, net, criterion, device)
print(acc_without_noise, acc_with_noise)

# only fine-tune certain layer
learning_rate_ft = 0.001
conv1_layer1 = chain(net.conv1.parameters(), net.layer1.parameters())
# optimizer_ft = optim.SGD(conv1_layer1, lr=learning_rate_ft, momentum=0.9, weight_decay=5e-4)
optimizer_ft = optim.SGD(net.parameters(), lr=learning_rate_ft, momentum=0.9, weight_decay=5e-4)
fine_tune_epochs = 50
early_stopping_patience = 30
min_val_loss = float('inf')

# store the loss values
train_losses = []
valid_losses = []

# train model
for epoch in range(fine_tune_epochs):
    print(f"Epoch {epoch+1}/{fine_tune_epochs}")
    train_loss, train_acc = train_epoch(subset_trainloader, net, criterion, optimizer_ft, device)
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
        torch.save(net.state_dict(), '/home/yushan/adaptive_personalization/motivation_exp/model/resnet26_model_ft_l3.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == early_stopping_patience:
            print('Early stopping!')
            break

print('Finished Training')


# load the best model
net.load_state_dict(torch.load('/home/yushan/adaptive_personalization/motivation_exp/model/resnet26_model_ft_l3.pth'))

# evaluation
test_loss, test_acc = test_epoch(testloader, net, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/home/yushan/adaptive_personalization/motivation_exp/fig/loss_single.png')


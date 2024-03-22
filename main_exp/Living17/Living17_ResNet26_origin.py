from robustness.tools.breeds_helpers import setup_breeds, print_dataset_info, ClassHierarchy, make_living17
from robustness.tools.vis_tools import show_image_row
from robustness import datasets
import matplotlib
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.transforms import transforms
import torchvision.transforms as tr
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from robustbench.utils import load_model
import warnings
warnings.filterwarnings("ignore")
import json
import matplotlib.pyplot as plt
from ResNet import ResNetCifar  # make sure in the same folder


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# creates living17 dataset from imagenet
# https://github.com/MadryLab/BREEDS-Benchmarks/blob/master/Constructing%20BREEDS%20datasets.ipynb

# Generate Living-17
data_dir = '/home/yushan/imagenet'
info_dir = './imagenet_class_hierarchy/modified'
num_workers = 5
batch_size = 224
reload_flag = 1 # 0 is the 1st to generate data, 1 is for reloading the data.

if reload_flag == 0:
    if not (os.path.exists(info_dir) and len(os.listdir(info_dir))):
        print("downloading class hierarchy information into `info_dir`")
        setup_breeds(info_dir)

    hier = ClassHierarchy(info_dir)
    ret = make_living17(info_dir, split="rand")
    superclasses, subclass_split, label_map = ret

    print_dataset_info(superclasses, subclass_split, label_map, hier.LEAF_NUM_TO_NAME)

    train_subclasses, test_subclasses = subclass_split

    # Save index information to reload
    with open('train_subclasses.json', 'w') as f:
        json.dump(train_subclasses, f)

    with open('test_subclasses.json', 'w') as f:
        json.dump(test_subclasses, f)

elif reload_flag == 1:
    # Load index information
    with open('train_subclasses.json', 'r') as f:
        train_subclasses = json.load(f)

    with open('test_subclasses.json', 'r') as f:
        test_subclasses = json.load(f)

    # Generate DataLoaders
    dataset_source = datasets.CustomImageNet(data_dir, train_subclasses)
    loaders_source = dataset_source.make_loaders(num_workers, batch_size)
    train_loader_source, val_loader_source = loaders_source

    dataset_target = datasets.CustomImageNet(data_dir, test_subclasses)
    loaders_target = dataset_target.make_loaders(num_workers, batch_size)
    train_loader_target, val_loader_target = loaders_target

else:
    raise ValueError('Please input a correct reload_flag: 0 or 1')


# # dataset size
# train_dataset_size_target = len(train_loader_target.dataset)
# val_dataset_size_target = len(val_loader_target.dataset)

# print(f"Train dataset (target) size: {train_dataset_size_target}")
# print(f"Validation dataset (target) size: {val_dataset_size_target}")


# Train original model
# Load original model
target_out_features = 17
model = ResNetCifar(depth=26).to(device)
model.fc = nn.Linear(in_features=1024, out_features=target_out_features)
model.to(device)
# print(model)

# Define training para
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=5e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

# define training and evaluation function
def evaluate_accuracy(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train()
    return total_loss / len(data_loader), correct / total

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, train_full=False):
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        for i, data in enumerate(tqdm(train_loader), 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        val_loss, val_acc = evaluate_accuracy(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if train_full == True:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy() 

        scheduler.step()

    # save the best model
    if train_full == True:
        if best_model_state is not None:
            model_filename = f"ResNet26_origin_Living17_val_acc_{val_accuracies[-1]:.4f}.pth"
            torch.save(best_model_state, model_filename)
            print(f"Best model saved as {model_filename}")
    
    return train_losses, val_losses, val_accuracies, model


epochs = 75

train_losses, val_losses, val_accuracies,_ = train_model(model, train_loader_source, val_loader_source, epochs, criterion, optimizer, train_full=True) 


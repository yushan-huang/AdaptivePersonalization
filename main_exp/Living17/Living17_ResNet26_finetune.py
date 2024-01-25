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
from itertools import chain
from torch.utils.data import Subset
import numpy as np
from ResNet import ResNetCifar  # make sure in the same folder



device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Generate Living-17
data_dir = '/home/yushan/imagenet'
info_dir = './imagenet_class_hierarchy/modified'
num_workers = 8
batch_size = 128
reload_flag = 1 # 0 is the 1st to generate data, 1 is for reloading the data.

if reload_flag == 0:
    print('1st generate data')
    if not (os.path.exists(info_dir) and len(os.listdir(info_dir))):
        print("downloading class hierarchy information into `info_dir`")
        setup_breeds(info_dir)

    hier = ClassHierarchy(info_dir)
    ret = make_living17(info_dir, split="rand")
    superclasses, subclass_split, label_map = ret

    print_dataset_info(superclasses, subclass_split, label_map, hier.LEAF_NUM_TO_NAME)

    train_subclasses, test_subclasses = subclass_split

    # Save index information to reload
    with open('/home/yushan/adaptive_personalization/main_exp/dataset/Living17/train_subclasses.json', 'w') as f:
        json.dump(train_subclasses, f)

    with open('/home/yushan/adaptive_personalization/main_exp/dataset/Living17/test_subclasses.json', 'w') as f:
        json.dump(test_subclasses, f)

elif reload_flag == 1:
    print('reload generated data')
    # Load index information
    with open('/home/yushan/adaptive_personalization/main_exp/dataset/Living17/train_subclasses.json', 'r') as f:
        train_subclasses = json.load(f)

    with open('/home/yushan/adaptive_personalization/main_exp/dataset/Living17/test_subclasses.json', 'r') as f:
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


# define optimizer
def part_finetune_optimizer(net, finetune_part):
    if finetune_part == 'layer0_layer1':

        para_finetune = chain(
            net.conv1.parameters(), 
            net.layer1.parameters()
        )
        LR = 0.001
    elif finetune_part == 'layer2':
        para_finetune = net.layer2.parameters()
        LR = 0.001
    elif finetune_part == 'layer3':
        para_finetune = net.layer3.parameters()
        LR = 0.001
    elif finetune_part == 'fc':
        para_finetune = net.fc.parameters()
        LR = 0.005
    elif finetune_part == 'all':
        para_finetune = net.parameters()
        LR = 0.001
    elif finetune_part == 'layer2_layer3':
        para_finetune = chain(
            net.layer2.parameters(), 
            net.layer3.parameters()
        )
        LR = 0.001

    else:
        raise ValueError('Please input a correct name of the finetuned part: layer0_layer1, layer2, layer3, fc, all')

    return optim.Adam(para_finetune, lr=LR)


# define training and evaluation function
def evaluate_accuracy(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(data_loader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train()
    return total_loss / len(data_loader), correct / total


def train_model(model, train_loader, val_loader, epochs, criterion, optimizer):
    
    train_losses = []
    val_losses = []
    val_accuracies = []

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

    return train_losses, val_losses, val_accuracies


def creat_sub_training_dataset(dataset, num_class, num_samples_per_class=50):
    indices_file = '/home/yushan/adaptive_personalization/main_exp/dataset/Living17/finetune_sub_training_target_dataset.json' 

    # check if the index file has already existed
    if os.path.exists(indices_file):
        with open(indices_file, 'r') as f:
            class_indices = json.load(f)
    else:
        # collect index
        class_indices = [[] for _ in range(num_class)]
        for idx, (_, label) in enumerate(tqdm(dataset)):  
            class_indices[label].append(idx)

        # save index
        with open(indices_file, 'w') as f:
            json.dump(class_indices, f)

    # randomly select 50 images
    selected_indices = []
    for indices in tqdm(class_indices):
        if len(indices) >= num_samples_per_class:
            selected_indices.extend(np.random.choice(indices, num_samples_per_class, replace=False))
        else:
            selected_indices.extend(indices)

    return selected_indices


def plt_figure(part,train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Test Accuracy: {val_accuracies[-1]:.4f}')
    plt.legend()
    plt.savefig(f'/home/yushan/adaptive_personalization/main_exp/fig/loss_Living17_ResNet26_ft_{part}.png')


# Load original model
target_out_features = 17
model = ResNetCifar(depth=26).to(device)
model.fc = nn.Linear(in_features=1024, out_features=target_out_features)
model.load_state_dict(torch.load('/home/yushan/adaptive_personalization/main_exp/model/ResNet26_origin_Living17_val_acc_0.8406.pth'))
model.to(device)
# print(model)

# generate a subset training dataset
train_dataset = train_loader_target.dataset
selected_indices = creat_sub_training_dataset(train_dataset, num_class=target_out_features, num_samples_per_class=100)
subset_train_loader_dataset_target = Subset(train_dataset, selected_indices)
balanced_train_loader_target = DataLoader(subset_train_loader_dataset_target, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Define training para
epochs = 10
criterion = nn.CrossEntropyLoss()
# finetune_part = ['layer0_layer1','layer2','layer3','fc','all','layer2_layer3']
finetune_part = ['layer0_layer1']

# Test
val_loss, val_acc = evaluate_accuracy(model, val_loader_target, criterion)
print('Without Finetune:', val_acc)

for part in finetune_part:

    # Finetune
    optimizer = part_finetune_optimizer(model, finetune_part=part)
    print(f'Fine-tune the {part} layer')
    train_losses, val_losses, val_accuracies = train_model(model, balanced_train_loader_target, val_loader_target, epochs, criterion, optimizer) 
    plt_figure(part,train_losses, val_losses, val_accuracies)


'''
Origin    No     Layer0_Layer1     Layer2     Layer3     Layer2_Layer3     FC       Full
84.06    45.82       52.18         58.41      68.47         68.53         61.71     67.12


'''
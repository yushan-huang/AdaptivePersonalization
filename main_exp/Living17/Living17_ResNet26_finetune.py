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
from torch.utils.data import DataLoader, random_split,ConcatDataset
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
import copy


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def collect_class_indices(dataset, num_classes, indices_file):
    if os.path.exists(indices_file):
        with open(indices_file, 'r') as file:
            class_indices = json.load(file)
    else:
        class_indices = [[] for _ in range(num_classes)]
        for idx, (_, label) in enumerate(tqdm(dataset)):
            class_indices[label].append(idx)

        with open(indices_file, 'w') as file:
            json.dump(class_indices, file)

    return class_indices

def split_dataset(class_indices, dataset, train_ratio=0.7, val_ratio=0.15):

    train_indices, val_indices, test_indices = [], [], []

    for indices in class_indices:
        np.random.shuffle(indices)
        train_end = int(len(indices) * train_ratio)
        val_end = train_end + int(len(indices) * val_ratio)

        train_indices.extend(indices[:train_end])
        val_indices.extend(indices[train_end:val_end])
        test_indices.extend(indices[val_end:])

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    return train_subset, val_subset, test_subset

# Generate Living-17
data_dir = './imagenet'
info_dir = './imagenet_class_hierarchy/modified'
num_workers = 8
batch_size = 128
reload_flag = 1 # 0 is to generate data, 1 is for reloading the data.

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
    with open('train_subclasses.json', 'w') as f:
        json.dump(train_subclasses, f)

    with open('test_subclasses.json', 'w') as f:
        json.dump(test_subclasses, f)

elif reload_flag == 1:
    print('reload generated data')
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

    combined_dataset = ConcatDataset([train_loader_target.dataset, val_loader_target.dataset])

    indices_file = "full_indices.json" 
    num_classes = 17  
    class_indices = collect_class_indices(combined_dataset, num_classes, indices_file)

    train_subset, val_subset, test_subset = split_dataset(class_indices, combined_dataset)

    train_loader_target = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader_target = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader_target = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


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
        LR = 0.01
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


def evaluate_accuracy(model, data_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):  
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy



def train_model(model, train_loader, val_loader, test_loader, epochs, criterion, optimizer, part,log_file):
    
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    best_model = None


    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        for i, data in enumerate(tqdm(train_loader, 0)):
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

        # check the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Evaluate the best model on the test dataset
    test_loss, test_acc = evaluate_accuracy(best_model, test_loader, criterion)
    final_message = f"Best Model {part} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
    print(final_message)
    with open(log_file, "a") as file:
        file.write(final_message + "\n")


    return train_losses, val_losses, val_accuracies



def creat_sub_training_dataset(dataset, num_class, num_samples_per_class=50) -> list:

    indices_file = 'finetune_sub_training_target_dataset.json'

    if os.path.exists(indices_file):
        with open(indices_file, 'r') as f:
            class_indices = json.load(f)
    else:
        class_indices = [[] for _ in range(num_class)]
        for idx, (_, label) in enumerate(tqdm(dataset)):
            if label < num_class: 
                class_indices[label].append(idx)
        
        with open(indices_file, 'w') as f:
            json.dump(class_indices, f)

    selected_indices = []
    for indices in class_indices:
        if len(indices) >= num_samples_per_class:
            selected_indices.extend(np.random.choice(indices, num_samples_per_class, replace=False))
        else:
            selected_indices.extend(indices)

    assert max(selected_indices) < len(dataset), "Selected index out of range."

    return selected_indices



finetune_part = ['layer0_layer1','layer2','layer3','fc','all']
num_per_class = [540] # According to training size

for num in num_per_class:
    for part in finetune_part:
        for i in range(4):

            # Load original model
            log_file = f"living17_result_{num}.txt" 
            target_out_features = 17
            model = ResNetCifar(depth=26).to(device)
            model.fc = nn.Linear(in_features=1024, out_features=target_out_features)
            model.load_state_dict(torch.load('ResNet26_origin_Living17.pth'))
            model.to(device)

            print(f'Num {num}, Part {part}, Time:{i+1}')
            # generate a subset training dataset
            train_dataset = train_loader_target.dataset
            selected_indices = creat_sub_training_dataset(train_dataset, num_class=target_out_features, num_samples_per_class=num)
            subset_train_loader_dataset_target = Subset(train_dataset, selected_indices)
            balanced_train_loader_target = DataLoader(subset_train_loader_dataset_target, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            # Define training para
            epochs = 10
            criterion = nn.CrossEntropyLoss()

            # Finetune
            optimizer = part_finetune_optimizer(model, finetune_part=part)
            train_losses, val_losses, val_accuracies = train_model(model, balanced_train_loader_target, val_loader_target, test_loader_target, epochs, criterion, optimizer, part, log_file)

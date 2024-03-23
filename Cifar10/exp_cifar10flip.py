from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset, Dataset, TensorDataset
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import train_test_split
from robustbench.model_zoo.enums import BenchmarkDataset
from typing import Dict, Optional, Sequence
from itertools import chain
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torchvision
import torch
import argparse
import random
import numpy as np
import pickle
import csv
import os

from ResNet import ResNetCifar

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

def _load_dataset(dataset, n_examples):
    batch_size = 100
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)
    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]
    return x_test_tensor, y_test_tensor   

def load_cifar10(n_examples, data_dir):
    dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transforms.ToTensor(), download=True)
    return _load_dataset(dataset, n_examples)  

def flip(target):
    class_mapping = {0: 9, 1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1, 9: 0}
    flipped_target = torch.tensor([class_mapping[class_item.item()] for class_item in target])
    return flipped_target

def split_dataset_by_class(x, y, train_size_ratio, val_size=0.15, test_size=0.15):
    x_remaining, x_test_val, y_remaining, y_test_val = train_test_split(x, y, test_size=test_size+val_size, random_state=42)
    val_size_adjusted = val_size / (test_size+val_size)  
    x_test, x_val, y_test, y_val = train_test_split(x_test_val, y_test_val, test_size=val_size_adjusted, random_state=42)
    train_size_final = train_size_ratio / (1 - val_size - test_size)
    x_train, _, y_train, _ = train_test_split(x_remaining, y_remaining, train_size=train_size_final, random_state=42)
    return x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--net_path', type=str, default='./resnets/resnet26_cifar10.pth')
    parser.add_argument('--ft_net_path', type=str, default='./resnets/resnet26_cifar10flip.pth')
    parser.add_argument('--cifar_path', type=str, default="./cifar/cifar-10-batches-py/")
    parser.add_argument('--results_path', type=str, default="./results/resnet26_cifar10flip.csv")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--ft_epochs', type=int, default=10)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--ts_ratio', type=float, default=0.1)
    args = parser.parse_args()

    net = ResNetCifar(depth=26).to(device)
    net.load_state_dict(torch.load(args.net_path, map_location=torch.device(device)))
    print(net)

    criterion = nn.CrossEntropyLoss()
    NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    norm = transforms.Normalize(*NORM)

    x, y = load_cifar10(10000, args.cifar_path) 
    x = norm(x)
    y_flipped = flip(y)
    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset_by_class(x, y_flipped, train_size_ratio=args.ts_ratio)
    train_ft = TensorDataset(x_train, y_train)
    val_ft = TensorDataset(x_val, y_val)
    test_ft = TensorDataset(x_test, y_test)
    train_loader_ft = DataLoader(train_ft, batch_size=args.batch_size, shuffle=True)
    val_loader_ft = DataLoader(val_ft, batch_size=args.batch_size)
    test_loader_ft = DataLoader(test_ft, batch_size=args.batch_size)

    conv1_block1 = chain(net.conv1.parameters(), net.layer1.parameters())
    layer_params = [conv1_block1,
                    net.layer2.parameters(), 
                    net.layer3.parameters(), 
                    net.fc.parameters(), 
                    net.parameters()]
    layer_params_txt = ["layer0_layer1", "layer2", "layer3", "fc", "all"]

    multi_lr, ct = False, 0
    layer_test_accs = {}
    for params in layer_params:
       
        net.load_state_dict(torch.load(args.net_path, map_location=torch.device(device))) 
        
        print(f"tuning {layer_params_txt[ct]}")
        
        if ct==len(layer_params)-2:
            learning_rate_ft = 1e-2
        else:
            learning_rate_ft = 1e-3
        
        optimizer_ft = torch.optim.SGD(params, lr=learning_rate_ft, momentum=0.9, weight_decay=5e-4)
        if multi_lr:
            scheduler = MultiStepLR(optimizer_ft, [5,8], gamma=0.1, last_epoch=-1, verbose=False)
        
        min_val_loss = float('inf')
        train_losses, valid_losses = [], []
        epochs_no_improve = 0
        for epoch in range(args.ft_epochs):
            print(f"epoch {epoch+1}/{args.ft_epochs}")
            train_loss, train_acc = train_epoch(train_loader_ft, net, criterion, optimizer_ft, device)
            valid_loss, valid_acc = test_epoch(val_loader_ft, net, criterion, device)
            print(f"train loss: {train_loss:.4f}, train acc: {train_acc:.2f}%")
            print(f"validation loss: {valid_loss:.4f}, validation acc: {valid_acc:.2f}%")
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            if valid_loss < min_val_loss:
                min_val_loss = valid_loss
                epochs_no_improve = 0
                torch.save(net.state_dict(), args.ft_net_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve == args.early_stopping_patience:
                    print('early stopping!')
                    break
            if multi_lr:
                scheduler.step()
       
        net.load_state_dict(torch.load(args.ft_net_path, map_location=torch.device(device)))
        test_loss, test_acc = test_epoch(test_loader_ft, net, criterion, device)
        print(f"test loss: {test_loss:.4f}, test acc: {test_acc:.2f}%")
        layer_test_accs[layer_params_txt[ct]] = test_acc
        ct += 1

    with open(args.results_path, 'w') as f:
        w = csv.DictWriter(f, layer_test_accs.keys())
        w.writeheader()
        w.writerow(layer_test_accs)

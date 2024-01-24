from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset, Dataset, TensorDataset
from torch.optim.lr_scheduler import MultiStepLR
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from typing import Dict, Optional, Sequence
from ResNet2 import ResNetCifar
from itertools import chain
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torchvision
import torch
import random
import numpy as np
import pickle
import csv
import os
from robustbench.utils import load_model
from sklearn.model_selection import train_test_split
from MobileNetV3 import MobileNetV3V  # make sure in the same folder



CORRUPTIONS = ("shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise", "defocus_blur", "brightness", 
               "fog", "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast", "jpeg_compression", "elastic_transform")


CORRUPTIONS_DIR_NAMES: Dict[BenchmarkDataset, str] = {BenchmarkDataset.cifar_10: {ThreatModel.corruptions: "CIFAR-10-C"}}

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

def load_cifar10c(data_dir, n_examples: int, severity: int = 5, shuffle = False, corruptions: Sequence[str] = CORRUPTIONS):
    return load_corruptions_cifar(data_dir, BenchmarkDataset.cifar_10, n_examples, severity, corruptions, shuffle)

def load_corruptions_cifar(data_dir, dataset: BenchmarkDataset, n_examples: int, severity: int, corruptions: Sequence[str] = CORRUPTIONS, shuffle = False):
    assert 1 <= severity <= 5
    n_total_cifar = 10000
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_dir = Path(data_dir)
    data_root_dir = data_dir / "CIFAR-10-C"
    labels_path = data_root_dir / 'labels.npy'
    labels = np.load(labels_path)
    x_test_list, y_test_list = [], []
    n_pert = len(corruptions)
    for corruption in corruptions:
        corruption_file_path = data_root_dir / (corruption + '.npy')
        images_all = np.load(corruption_file_path)
        images = images_all[(severity - 1) * n_total_cifar:severity * n_total_cifar]
        n_img = int(np.ceil(n_examples / n_pert))
        x_test_list.append(images[:n_img])
        y_test_list.append(labels[:n_img])
    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
    if shuffle:
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    x_test = x_test.astype(np.float32) / 255
    x_test = torch.tensor(x_test)[:n_examples]
    y_test = torch.tensor(y_test)[:n_examples]
    return x_test, y_test 



def split_dataset_by_class(x, y, train_size_ratio, val_size=0.15, test_size=0.15):
    x_remaining, x_test_val, y_remaining, y_test_val = train_test_split(x, y, test_size=test_size+val_size, random_state=42)
    
    val_size_adjusted = val_size / (test_size+val_size)  
    x_test, x_val, y_test, y_val = train_test_split(x_test_val, y_test_val, test_size=val_size_adjusted, random_state=42)

    train_size_final = train_size_ratio / (1 - val_size - test_size)
    x_train, _, y_train, _ = train_test_split(x_remaining, y_remaining, train_size=train_size_final, random_state=42)

    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # net = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')
    # torch.save(net.state_dict(), "models/cifar10/Linf/Standard.pt")
    # net = net.to(device)
    # print(net)

    # net = MobileNetV3V(num_classes=10).to(device)
    # net.load_state_dict(torch.load('/home/yushan/adaptive_personalization/motivation_exp/model/mobilenet_model.pth'))

    net = ResNetCifar(depth=26).to(device)
    net.load_state_dict(torch.load('/home/yushan/adaptive_personalization/main_exp/model/ResNet26_origin_Living17_val_acc_0.8406.pth'))

    batch_size = 128
    criterion = nn.CrossEntropyLoss()
    NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    norm = transforms.Normalize(*NORM)

    # this averages acc + loss across CORRUPTIONS
    corruption_test_accs = {}
    for corruption in tqdm(CORRUPTIONS):
        print(f"corruption = {corruption}")
        x_corrupt, y_corrupt = load_cifar10c("../cifar/", 10000, 5, False, [corruption])
        x_corrupt = norm(x_corrupt)

        # Yushan revise start
        x_train, y_train, x_val, y_val, x_test, y_test = split_dataset_by_class(x_corrupt, y_corrupt, train_size_ratio=0.2)
    
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        test_dataset = TensorDataset(x_test, y_test)

        train_loader_ft = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader_ft = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader_ft = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # Yushan revise end

        # all_indices = list(range(len(x_corrupt)))
        # random.shuffle(all_indices)
        # train_size = 1000
        # train_indices = all_indices[:train_size]
        # val_indices = all_indices[train_size:train_size+train_size//10]
        # test_indices = all_indices[2*train_size:5*train_size]

        # train_ft = TensorDataset(x_corrupt[train_indices], y_corrupt[train_indices])
        # val_ft = TensorDataset(x_corrupt[val_indices], y_corrupt[val_indices])
        # test_ft = TensorDataset(x_corrupt[test_indices], y_corrupt[test_indices])

        # train_loader_ft = DataLoader(train_ft, batch_size=batch_size)
        # val_loader_ft = DataLoader(val_ft, batch_size=batch_size)
        # test_loader_ft = DataLoader(test_ft, batch_size=batch_size)


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
            # net.load_state_dict(torch.load('./models/cifar10/Linf/Standard.pt'))
            net.load_state_dict(torch.load('/home/yushan/adaptive_personalization/main_exp/model/ResNet26_origin_Living17_val_acc_0.8406.pth'))
            print(f"tuning {layer_params_txt[ct]}")
            if ct==len(layer_params)-2:
                learning_rate_ft = 1e-2
            else:
                learning_rate_ft = 1e-3
            # optimizer_ft = torch.optim.SGD(params, lr=learning_rate_ft, momentum=0.9, weight_decay=5e-4)
            optimizer_ft = torch.optim.SGD(params, lr=learning_rate_ft)
            if multi_lr:
                scheduler = MultiStepLR(optimizer_ft, [5,8], gamma=0.1, last_epoch=-1, verbose=False)
            fine_tune_epochs = 20
            early_stopping_patience = 10
            min_val_loss = float('inf')
            train_losses, valid_losses = [], []
            epochs_no_improve = 0
            for epoch in range(fine_tune_epochs):
                print(f"epoch {epoch+1}/{fine_tune_epochs}")
                train_loss, train_acc = train_epoch(train_loader_ft, net, criterion, optimizer_ft, device)
                valid_loss, valid_acc = test_epoch(val_loader_ft, net, criterion, device)
                print(f"train loss: {train_loss:.4f}, train acc: {train_acc:.2f}%")
                print(f"validation loss: {valid_loss:.4f}, validation acc: {valid_acc:.2f}%")
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                if valid_loss < min_val_loss:
                    min_val_loss = valid_loss
                    epochs_no_improve = 0
                    torch.save(net.state_dict(), './resnets/resnet_cifar.pth')
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == early_stopping_patience:
                        print('early stopping!')
                        break
                if multi_lr:
                    scheduler.step()
            net.load_state_dict(torch.load('./resnets/resnet_cifar.pth', map_location=torch.device('cpu')))
            test_loss, test_acc = test_epoch(test_loader_ft, net, criterion, device)
            print(f"test loss: {test_loss:.4f}, test acc: {test_acc:.2f}%")
            layer_test_accs[layer_params_txt[ct]] = test_acc
            ct += 1
        print(layer_test_accs)
        corruption_test_accs[corruption] = layer_test_accs
    print(corruption_test_accs)
    corruption_avg_layer_test_accs = {}
    for layer in layer_params_txt:
        accs = []
        for corruption in CORRUPTIONS:
            accs.append(corruption_test_accs[corruption][layer])
        corruption_avg_layer_test_accs[layer] = np.mean(accs)
    print(corruption_avg_layer_test_accs)
    with open('results/results_resnet_cifar.csv', 'w') as f:
        w = csv.DictWriter(f, corruption_avg_layer_test_accs.keys())
        w.writeheader()
        w.writerow(corruption_avg_layer_test_accs)
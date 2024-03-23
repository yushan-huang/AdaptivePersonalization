import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import time

from ResNet import ResNetCifar

start = time.time()

target_out_features = 17 # Cifar10-C, Cifar-Flip: 10, Living17: 17
model = ResNetCifar(depth=26)
model.fc = nn.Linear(in_features=64, out_features=target_out_features) # input 32x32, in_features=64; input 96x96, in_features=1024

random_data = torch.randn(1, 3, 32, 32) # input size 32x32/96x96，100 samples
random_labels = torch.randint(0, 10, (1,)).long() # Cifar10-C, Cifar-Flip: 10, Living17: 17

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

train_data = torch.utils.data.TensorDataset(random_data, random_labels)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)

def train_partial_model(model, train_loader, optimizer, criterion, layers_to_train): 
   for name, param in model.named_parameters():
       param.requires_grad = any(layer in name for layer in layers_to_train)
   print(f"started ft at {time.time()-start}")
   model.train()
   for epoch in range(1):
       for data, target in train_loader:
           optimizer.zero_grad()
           output = model(data)
           loss = criterion(output, target)
           loss.backward()
           optimizer.step()
   print(f"ended ft at {time.time()-start}")

# train certain layers
# [‘conv1’,‘layer1’]
# [‘layer2’,‘layer3’]
# [‘layer2’]
# [‘layer3’]
# [‘fc’]
# [‘full’]
layers = ["fc"]
train_partial_model(model, train_loader, optimizer, criterion, layers)

time.sleep(30)

train_partial_model(model, train_loader, optimizer, criterion, [name for name, _ in model.named_parameters()]) # full

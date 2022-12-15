# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# import torch lib
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
from torchvision.datasets import DatasetFolder, ImageFolder
from torchsampler import ImbalancedDatasetSampler
# envalid lib
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("train_folder")
parser.add_argument("valid_folder")
args = parser.parse_args()

def fix_seeds(seed):
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# global variable.
seed = 100
fix_seeds(seed)
name = args.name 
input_size = 224 
trainset_root = args.train_folder 
validset_root = args.valid_folder
#trainset_root = "~/Documents/class/dlcv/final-project-challenge-3-so_ez_peasy/food_data/train"
#validset_root = "~/Documents/class/dlcv/final-project-challenge-3-so_ez_peasy/food_data/val"
num_workers = 4 
do_trans_learn = False

# hyperparameters.
batch_size = 64 
n_epoch = 31 
lr = 0.001
lr_step_size = 25
lr_gamma = 0.5
l2_lambda = 0 # regulization.
max_norm = 7 # gradient clipping.
alpha = 1.0 # mixup.

# load check_point.
load_ckpt = False 
ckpt_load_path = "./ckpt/41ba478/best.pt"
ckpt_root = f"./ckpt/{name}"
os.makedirs(ckpt_root, exist_ok=True)
ckpt_best_path = os.path.join(ckpt_root, "best.pt") 
last_epoch = 0
best_acc = 0

# log file.
log_root = f"./log/{name}"
os.makedirs(log_root, exist_ok=True)
log_file = os.path.join(log_root, "log")
train_log_csv = os.path.join(log_root, "train.csv")
valid_log_csv = os.path.join(log_root, "valid.csv")
with open(train_log_csv, "a") as f:
    f.write("loss,acc\n")
with open(valid_log_csv, "a") as f:
    f.write("loss,acc\n")
    
# device.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Transform definition train_transform & valid_transform.
trainset_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ColorJitter(),
    transforms.GaussianBlur(31),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomRotation(degrees=45),
    transforms.ToTensor(),
])
validset_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
])

# model initalize.
def set_model_grad(model, do_trans_learn):
  if do_trans_learn:
    for param in model.parameters():
      param.requires_grad = False

def init_model_resnet(num_classes=1000, do_trans_learn=False, use_pretrained=True):
  model = models.resnet101(pretrained=use_pretrained)
  set_model_grad(model, do_trans_learn)
  last_fc_input_size = model.fc.in_features
  model.fc = nn.Linear(last_fc_input_size, num_classes)
  return model

# training function.
def train(model, dataloader, loss_func, optimizer):
  model.train() 
  losses = []
  accs = []

  for inputs, targets in tqdm(dataloader):
    # load to device.
    inputs = inputs.to(device) 
    targets = targets.to(device)

    # compute output and loss. 
    # loss in default will be divided by num element in a batch.
    # loss, acc = normal_train(model, loss_func, inputs, targets)
    # loss, acc = input_mix_up_train(model, loss_func, inputs, targets)
    loss, acc = mm_mix_up_train(model, loss_func, inputs, targets)
    # update model.
    optimizer.zero_grad()
    loss.backward()
    # gradient clipping.
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()
    
    # calculate loss and acc.
    losses.append(loss.item())
    accs.append(acc.item())

  avg_loss = np.array(losses).mean()
  avg_acc = np.array(accs).mean()

  return avg_loss, avg_acc

def normal_train(model, loss_func, inputs, targets):
  outputs = model(inputs)
  loss = loss_func(outputs, targets)
  acc = (outputs.argmax(dim=-1)==targets).float().mean()
  return loss, acc

def input_mix_up_train(model, loss_func, inputs, targets):
  l = np.random.beta(alpha, alpha)
  idx = torch.randperm(inputs.size(0))
  image_a, image_b = inputs, inputs[idx]
  label_a, label_b = targets, targets[idx]
  mixed_image = l * image_a + (1-l) * image_b
  
  outputs = model(mixed_image)
  loss = l * loss_func(outputs, label_a) + (1-l) * loss_func(outputs, label_b)
  result = outputs.argmax(dim=-1)
  acc = l * (result == label_a).float().mean() + (1-l) * (result == label_b).float().mean()
  return loss, acc

def mm_mix_up_train(model, loss_func, inputs, targets):
  l = np.random.beta(alpha, alpha)
  idx = torch.randperm(inputs.size(0))
  image_a, image_b = inputs, inputs[idx]
  label_a, label_b = targets, targets[idx]

  # thrid last layer : torch.nn.modules.container.Sequential
  # second last layer : torch.nn.modules.pooling.AdaptiveAvgPool2d
  if_mixed = False
  mixed_image = None
  for layer in model.children():
    if not if_mixed:
      if isinstance(layer, torch.nn.modules.linear.Linear):
        mixed_image = l * image_a + (1-l) * image_b
        mixed_image = mixed_image.squeeze()
        mixed_image = layer(mixed_image)
        if_mixed = True
        continue
      image_a = layer(image_a)
      image_b = layer(image_b)
    else:
      assert mixed_image != None
      mixed_image = layer(mixed_image) 
  outputs = mixed_image
  loss = l * loss_func(outputs, label_a) + (1-l) * loss_func(outputs, label_b)
  result = outputs.argmax(dim=-1)
  acc = l * (result == label_a).float().mean() + (1-l) * (result == label_b).float().mean()
  return loss, acc

def valid(model, dataloader, loss_func, optimizer):
  model.eval()
  losses = []
  acces = []

  for inputs, targets in tqdm(dataloader):

    inputs = inputs.to(device)
    targets = targets.to(device)

    outputs = model(inputs)
    loss = loss_func(outputs, targets)

    losses.append(loss.item())
    acc = (outputs.argmax(dim=-1)==targets).float().mean()
    acces.append(acc.item())
  
  avg_loss = np.array(losses).mean()
  avg_acc = np.array(acces).mean()

  return avg_loss, avg_acc

def load_check_point(ckpt_path, model, optimizer):
  print("load check point.")
  global last_epoch
  global best_acc
  checkpoint = torch.load(ckpt_path)
  last_epoch = checkpoint['last_epoch'] + 1
  #best_acc = checkpoint['best_acc'] 
  model.load_state_dict(checkpoint['model_state_dict'])
  #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def save_check_point(ckpt_path, model, optimizer, last_epoch, best_acc):
  print(f"save check point at {last_epoch} epochs, acc = {best_acc}")
  torch.save({
    'last_epoch': last_epoch,
    'best_acc' : best_acc,
    'model_state_dict' : model.state_dict()
 #   'optimizer_state_dict' : optimizer.state_dict()
  }, ckpt_path)

# dataset preparing.
# train_dataset = DatasetFolder(trainset_root, loader=lambda x: Image.open(x), extensions="jpg", transform=trainset_transform)
# valid_dataset = DatasetFolder(validset_root, loader=lambda x: Image.open(x), extensions="jpg", transform=validset_transform)
train_dataset = ImageFolder(trainset_root, transform=trainset_transform)
valid_dataset = ImageFolder(validset_root, transform=validset_transform)
train_loader = DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size=batch_size, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# model.
model = init_model_resnet(num_classes=1000, do_trans_learn=False).to(device)

# loss function.
loss_func = nn.CrossEntropyLoss() 

# optimizer.
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)

# load checkpoint.
if load_ckpt:
  load_check_point(ckpt_load_path, model, optimizer)

# lr scheduling.
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

train_losses = []
valid_losses = []
# training.
for epoch in range(last_epoch, n_epoch):

  loss, acc = train(model, train_loader, loss_func, optimizer)
  train_losses.append(loss)
  print(f"Training {epoch}/{n_epoch}, loss : {loss}, acc : {acc}")
  with open(train_log_csv, "a") as f:
      f.write(f"{loss},{acc}\n")

  loss, acc = valid(model, valid_loader, loss_func, optimizer)
  valid_losses.append(loss)
  print(f"Validate {epoch}/{n_epoch}, loss : {loss}, acc : {acc}")
  with open(valid_log_csv, "a") as f:
      f.write(f"{loss},{acc}\n")

  #scheduler.step()
  if acc >= best_acc:
    best_acc = acc
    save_check_point(ckpt_best_path, model, optimizer, epoch, best_acc)
    with open(log_file, "a") as f:
        f.write(f"best acc in epoch {epoch}, acc = {best_acc}\n")
    if acc >= 0.7:
        acc_str = str(acc)[:5]
        ckpt_path = os.path.join(ckpt_root, f"{acc_str}_{epoch}.pt")
        save_check_point(ckpt_path, model, optimizer, epoch, best_acc)

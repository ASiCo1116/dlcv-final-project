import time
import numpy as np
import sys
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type(torch.FloatTensor)

data_path = sys.argv[1]         # data path
batch_size = sys.argv[2]        # batch size
num_epoch = sys.argv[3]         # number of epoch
saved_model_path = sys.argv[4]  # saved model path

train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

val_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(data_path + '/train', transform=train_transform)
val_data = datasets.ImageFolder(data_path + '/val', transform=val_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

print('len trainloader', len(train_loader.dataset))
print('len batch trainloader', len(train_loader))

model =models.densenet121(pretrained=True)
model.to(device)

for param in model.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(num_epochs,train_loader,val_loader, backbone, optimizer, criterion):
    writer = SummaryWriter()
    start = time.time()
    val_loss_min = np.Inf
    step = 0
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        train_loss=0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = backbone(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, argmax = torch.max(outputs, 1)
            accuracy = (labels == argmax.squeeze()).float().mean()

            if (i+1) %100 == 0:
                time_cost = time.time()-start
                print ('Train: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {}, Time:{}'.format(epoch+1, num_epochs, i+1, total_step, loss.item(), accuracy.item(), time_cost))

                writer.add_scalar('Loss/train', loss.item(), step)
                writer.add_scalar('Accuracy/train', accuracy, step)
                step += 100
        torch.save(model.state_dict(), f'{saved_model_path}/classifier_{time.time()}.pth')

        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                images, labels = data[0].to(device), data[1].to(device)

                outputs = model(images)
                batch_loss = criterion(outputs, labels)
                val_loss += batch_loss.item()

                top_p, top_class = outputs.topk(5, dim=1)
                labels = labels.view(-1,1)   
                results =  top_class == labels
                results = results.sum(1)
                val_accuracy += torch.mean(results.float()).item()

        avg_val_accuracy = val_accuracy/(i+1)
        avg_val_loss = val_loss/(i+1)
        time_cost = time.time()-start
        print ('Test: Epoch [{}/{}], Loss: {:.4f}, Accuracy Top5: {:.3%}, Time:{}'.format(epoch+1, num_epochs, avg_val_loss, avg_val_accuracy, time_cost))
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', avg_val_accuracy, epoch)

        if val_loss <= val_loss_min:
            torch.save(model.state_dict(),f'{saved_model_path}/classifier_lowest_loss.pth')
            val_loss_min=avg_val_loss

        model.train()

train(num_epoch, train_loader, val_loader, model, optimizer, criterion)



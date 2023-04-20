import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from lenet import LeNet_add
from lenet_cnn import LeNet
import wandb
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='train-addernet')

# Basic model parameters.
parser.add_argument('--data', type=str, default='./cache/data/')
parser.add_argument('--output_dir', type=str, default='./cache/models/')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--mode', type=str, default='ann')
parser.add_argument('--name', type=str, default='ANN-lenet5-mnist')
parser.add_argument('--epoch', type=int, default=400)
args = parser.parse_args()


device = args.device

train_data = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor(),
                      transforms.Normalize((0.5,), (0.2,))]))
test_data = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5,), (0.2,))]))

train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=256, num_workers=2)

if args.mode == 'cnn':
    net = LeNet().to(device)
elif args.mode == 'ann':
    net = LeNet_add().to(device)
else:
    raise Exception("Invalid mode")
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
NEPOCHE = args.epoch
NITER = NEPOCHE * len(train_loader)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15], gamma=0.5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NITER)

def train(epoch):
    # wandb.watch(net, criterion, log='all', log_freq=10)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images, labels = images.to(device), labels.to(device)
 
        optimizer.zero_grad()
 
        output = net(images)
 
        loss = criterion(output, labels)
 
        train_loss += loss.data.item()
 
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        _, predicted = output.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
    print('Train - Epoch %d, Loss: %f, Acc :%f' % (epoch, train_loss / (i+1), 100.*correct/total))
    # wandb.log({'train':{'Train_Loss':train_loss/(i+1),
    #                     'Train_Acc':100.*correct/total}},
    #           step=epoch)


def test(epoch):
    global acc, acc_best
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            output = net(images)
            test_loss += criterion(output, labels).sum()
            
            _, predicted = output.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
 
    print('Test - Epoch %d, Loss: %f, Acc :%f' % (epoch, test_loss / (i+1), 100.*correct/total))
    # wandb.log({'test':{'Test_Loss':test_loss/(i+1),
    #                    'Test_Acc':100.*correct/total}},
    #           step=epoch)


def train_and_test(epoch):
    train(epoch)
    test(epoch)

def main():
    # wandb.init(project='Addernet', entity='kwonrince', name='ANN-LeNet5-MNIST')
    for e in range(1, NEPOCHE):
        train_and_test(e)
    # wandb.finish()

if __name__ == '__main__':
    main()

import os
from resnet20 import resnet20_add
from resnet20_cnn import resnet20
import torch
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
import argparse
import math
from tqdm import tqdm
# import wandb

parser = argparse.ArgumentParser(description='train-addernet')

# Basic model parameters.
parser.add_argument('--data', type=str, default='./cache/data/')
parser.add_argument('--output_dir', type=str, default='./cache/models/')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--name', type=str, default='ANN-resnet20-cifar10')
parser.add_argument('--epoch', type=int, default=50)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

device = args.device
acc = 0
acc_best = 0

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

data_train = CIFAR10(args.data,
                   transform=transform_train,
                   download=True)
data_test = CIFAR10(args.data,
                  train=False,
                  transform=transform_test,
                  download=True)

data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=100, num_workers=0)

net = resnet20_add().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    lr = 0.05 * (1+math.cos(float(epoch)/10*math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train(epoch):
    # wandb.watch(net, criterion, log='all', log_freq=10)
    adjust_learning_rate(optimizer, epoch)
    global cur_batch_win
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(tqdm(data_train_loader)):
        images, labels = Variable(images).to(device), Variable(labels).to(device)
 
        optimizer.zero_grad()
 
        output = net(images)
 
        loss = criterion(output, labels)
 
        train_loss += loss.data.item()
 
        loss.backward()
        optimizer.step()
        
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
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).to(device), Variable(labels).to(device)
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
    # wandb.init(project='Addernet', entity='kwonrince', name=args.name)
    epoch = args.epoch
    for e in range(1, epoch):
        train_and_test(e)
    # wandb.finish()
 
 
if __name__ == '__main__':
    main()
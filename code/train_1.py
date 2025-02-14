import torch
import torchvision
from skimage import io
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dataset_1
import fcrn_structure
import matplotlib.pyplot as plt
import torchvision.utils as utils
from pathlib import Path
import numpy as np
from tensorboardX import SummaryWriter
from distutils.version import LooseVersion

writer = SummaryWriter("runs/5")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform4gray = transforms.Compose(
    [transforms.ToTensor()])

b_size = 4

trainset = dataset_1.LoadDataset1(transform=transform, transform4gray=transform4gray, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True)
testset = dataset_1.LoadDataset1(transform=transform, transform4gray=transform4gray, train=False)
testLoader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)




def train():
    # loss function and optimizer
    net = fcrn_structure.FCResnet50(pretrained=True)
    criterion = nn.CrossEntropyLoss()
    epochs = 20
    learning_rate = 0.001
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0005)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        total = 0.0
        correct = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.long()
            labels = torch.flatten(labels, start_dim=1, end_dim=2)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            # print(outputs)
            # print(labels)
            loss = criterion(outputs, labels)
            # print(i, ' ', loss)
            loss.backward()
            if epoch > 30:
                learning_rate = 0.000001
            elif epoch > 20:
                learning_rate = 0.00001
            elif epoch > 10:
                learning_rate = 0.0001
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # calculate acc
            op_labels = torch.flatten(labels, 0)
            op_labels = op_labels.long()
            total += op_labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            predicted = torch.flatten(predicted, 0)
            correct += (predicted == op_labels).sum().item()
            if i % 15 == 14:  # print every 60 mini-batches
                acc = correct / total
                print('[%d, %5d] loss: %.5f, acc: %.5f' %
                      (epoch + 1, i + 1 + len(trainloader) * epoch, running_loss / 15, acc))
                print('training progress: %.5f%%' % (
                        (i + 1 + len(trainloader) * epoch) / (len(trainloader) * epochs) * 100))
                writer.add_scalar("Train/Loss", running_loss / 15, i + 1 + len(trainloader) * epoch)
                writer.add_scalar("Train/Acc", acc, i + 1 + len(trainloader) * epoch)
                print()
                running_loss = 0.0
                total = 0.0
                correct = 0.0

    print('Finished Training')
    writer.close()

    PATH = Path('../Path/train1_v3.pth')
    torch.save(net.state_dict(), PATH)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def test():

    # print images
    #imshow(torchvision.utils.make_grid(images))
    #imshow(torchvision.utils.make_grid(labels))

    net = fcrn_structure.FCResnet50()
    PATH = Path('../Path/train1_v2.pth')
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    i = 0
    with torch.no_grad():
        for data in testLoader:
            print(i)
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            #imshow(torchvision.utils.make_grid(images))
            #imshow(torchvision.utils.make_grid(predicted))
            #imshow(torchvision.utils.make_grid(labels))
            predicted = torch.flatten(predicted, 0)
            labels = torch.flatten(labels, 0)
            labels = labels.long()
            total += labels.size(0)
            print('total: %d' % total)
            correct += (predicted == labels).sum().item()
            print('correct: %d' % correct)
            i = i + 1
            print(predicted)
            print(labels)
    print('Accuracy of the network on the 378 test images: %.3f %%' % (
            100 * correct / total))

def testforone(a):
    net = fcrn_structure.FCResnet50()
    PATH = Path('../Path/train1_v3.pth')
    net.load_state_dict(torch.load(PATH))
    i = 0
    with torch.no_grad():
        for data in testLoader:
            if i == a:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                imshow(torchvision.utils.make_grid(images))
                imshow(torchvision.utils.make_grid(predicted))
                imshow(torchvision.utils.make_grid(labels))
            i = i + 1

testforone(171)
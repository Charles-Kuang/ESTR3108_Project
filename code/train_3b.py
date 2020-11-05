import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils as utils
from skimage import io
from pathlib import Path
from tensorboardX import SummaryWriter
import numpy as np
import dataset_3b
import drn_structure


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#for flip, later we will use this
def h_flip(image):
    return cv2.flip(image, 1)
def v_flip(image):
    return cv2.flip(image, 0)

b_size = 16  

trainset = dataset_3b.LoadDataset3b(transform=transform, train=True)
testset = dataset_3b.LoadDataset3b(transform=transform, train=False)

def make_weights_for_balanced_classes(items, classes):
    count = [0] * classes
    for item in items:
        if(item[1] == 'benign'):
            count[0] = count[0] + 1
            item[1] = 0
        elif(item[1] == 'malignant'):
            count[1] = count[1] + 1
            item[1] = 1
    weight_per_class = [0.] * classes
    N = float(sum(count))
    for i in range(classes):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(items)
    for idx, item in enumerate(items):
        weight[idx] = weight_per_class[item[1]]
    return weight

weights = make_weights_for_balanced_classes(trainset.train_list, 2)
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, sampler=sampler, pin_memory=True)

testLoader = torch.utils.data.DataLoader(testset, batch_size=b_size, shuffle=True, pin_memory=True)


##train
def train():
    # loss function and optimizer
    net = drn_structure.resnet50(pretrained= True)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter("Res50")
    epochs = 30
    lr = 0.002
    
    PATH = Path('../Path/train3b_v1.pth')
    ##train
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0
        total = 0
        if epoch % 2 == 1:
            lr = lr / 5
        else:
            lr = lr / 2
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        #optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 5 == 4:  # print every 5 mini-batches
                running_loss = running_loss / 5
                accuracy = correct / total
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss), 'accuracy:' , accuracy)
                writer.add_scalar("Train/Loss", running_loss, epoch * 57 + i)
                writer.add_scalar("Train/Acc",accuracy, epoch * 57 + i)
                running_loss = 0.0
                correct = 0
                total = 0

        torch.save(net.state_dict(), PATH)

    print('Finished Training')

def test():
    net = drn_structure.resnet50(pretrained = False)
    PATH = Path('../Path/K_train3b_v2.pth')
    net.load_state_dict(torch.load(PATH))

    correct = 0
    correct1 = 0
    correct2 = 0
    total = 0
    total1 = 0
    total2 = 0

    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))

    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            print(predicted)
            print(labels)
            for i in range(labels.size(0)):
                label = int(labels[0].item())
                class_correct[label] += c[i].item()
                class_total[label] += 1
        for i in range(2):
            print('Accuracy of %5s : %2d %%' % (
            i, 100 * class_correct[i] / class_total[i]))
    print('Accuracy of the network on the' , total , 'test images: %d %%' % (
            100 * correct / total))

#train()
test()


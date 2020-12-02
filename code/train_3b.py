import cv2
import math
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
import width

device = torch.device("cuda:0")

transform_hf = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.RandomCrop(250, padding=0),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_vf = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.RandomCrop(250, padding=0),
    transforms.RandomVerticalFlip(p=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_hvf = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.RandomCrop(250, padding=0),
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomVerticalFlip(p=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#for flip, later we will use this
def h_flip(image):
    return cv2.flip(image, 1)
def v_flip(image):
    return cv2.flip(image, 0)

b_size = 16

trainset = dataset_3b.LoadDataset3b(transform=transform_hf, train=True)
"""
trainseth = dataset_3b.LoadDataset3b(transform=transform_hf, train=True)
trainsetv = dataset_3b.LoadDataset3b(transform=transform_vf, train=True)
trainsethv = dataset_3b.LoadDataset3b(transform=transform_hvf, train=True)
"""
testset = dataset_3b.LoadDataset3b(transform=transform, train=False)

def make_weights_for_balanced_classes(items, classes):
    count = [0] * classes
    for item in items:
        if(item[1] == 'benign' or item[1]==0):
            count[0] = count[0] + 1
            item[1] = 0
        elif(item[1] == 'malignant' or item[1]==1):
            count[1] = count[1] + 1
            item[1] = 1
    weight_per_class = [0.] * classes
    N = float(sum(count))
    for i in range(classes):
        weight_per_class[i] = (N/2) / float(count[i])
    #weight_per_class[0] = weight_per_class[0] * 2
    weight = [0] * len(items)
    for idx, item in enumerate(items):
        weight[idx] = weight_per_class[item[1]]
    return weight, weight_per_class
weights, weight_per_class = make_weights_for_balanced_classes(trainset.train_list, 2)
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, sampler=sampler, pin_memory=True)

testLoader = torch.utils.data.DataLoader(testset, batch_size=b_size, shuffle=False, pin_memory=True)



##train
def train():
    # loss function and optimizer
    net = width.wide_resnet50_2(pretrained = True)
    #net.load_state_dict(torch.load(PATH))
    print(weight_per_class)
    
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter("Res50/16B-test")
    
    epochs = 50
    lr = 0.01
    t = 0
    ##train
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        if (epoch % 7 == 0 and lr>=1e-05):
            lr = lr / 10
        print(lr)
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward 
            outputs = net(inputs).to(device)
            loss = criterion(outputs, labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            del outputs
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 10 == 9:  # print every 5 mini-batches
                running_loss = running_loss / 10
                accuracy = correct / total
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss), 'accuracy:' , accuracy)
                writer.add_scalar("Train/Loss", running_loss, t)
                writer.add_scalar("Train/Acc",accuracy, t)
                t+=1
                running_loss = 0.0
                correct = 0
                total = 0
                threshold = 1
                if t % 2 == 0 and t > 0:
                    PATH = Path('../Path/16B-test/' + str(epoch+1) + '-' + str(t) + '.pth')
                    torch.save(net.state_dict(), PATH)
                    test(PATH, t, threshold)
    print('Finished Training')



def test(PATH, t, threshold):
    tnet = drn_structure.resnet50(pretrained = True)
    tnet.load_state_dict(torch.load(PATH))
    tnet.eval()

    writer_t_0 = SummaryWriter("Res50/0")
    writer_t_1 = SummaryWriter("Res50/1")
    
    correct = 0
    correct1 = 0
    correct2 = 0
    total = 0
    total1 = 0
    total2 = 0

    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    print(threshold)
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            images = images.to(device)
            labels =labels.to(device)
            outputs = tnet(images).to(device)
            
            i = 0
            for output in outputs:
                outputs[i][0] = outputs[i][0] / threshold
                i += 1
            
            _, predicted = torch.max(outputs.data, 1)
            


            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = int(labels[i].item())
                #print(labels[0].item())
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(2):
        print('Accuracy of %5s : %.5f %%' % (
        i, 100 * float(class_correct[i]) / class_total[i]))
    writer_t_0.add_scalar("Test/0", float(class_correct[0]) / class_total[0], t)
    writer_t_1.add_scalar("Test/1", float(class_correct[1]) / class_total[1], t)            
    print('Accuracy of the network on the' , total, t , 'test images: %.5f %%' % (
            100 * float(correct) / total))
    
train()
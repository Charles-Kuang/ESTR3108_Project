import torch
import torchvision
from skimage import io
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils as utils
from pathlib import Path
import numpy as np
import dataset_3b
import drn_structure

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

b_size = 16

trainset = dataset_3b.LoadDataset3b(transform=transform, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True)
testset = dataset_3b.LoadDataset3b(transform=transform, train=False)
testLoader = torch.utils.data.DataLoader(testset, batch_size=b_size, shuffle=True)


##train
def train():
    # loss function and optimizer
    net = drn_structure.resnet50()
    criterion = nn.CrossEntropyLoss()
    epochs = 5

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.0005)

    ##train
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            #print(i, ' ', loss)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:  # print every 5 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

    print('Finished Training')

    PATH = Path('../Path/train3b_v1.pth')
    torch.save(net.state_dict(), PATH)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test():
    #dataiter = iter(testLoader)
    #images, labels = dataiter.next()

    # print images
    #imshow(torchvision.utils.make_grid(images))
    #print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(4)))

    net = drn_structure.resnet50()
    PATH = Path('../Path/train3b_v1.pth')
    net.load_state_dict(torch.load(PATH))

    print('Predicted: ', ' '.join('%5s' % labels[predicted[j]] for j in range(4)))

    correct = 0
    total = 0
    i = 0
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(i)
            i = i + 1
            print(predicted)
            print(labels)
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

train()
test()


import torch
import torchvision
from skimage import io
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import dataset_3
import drn_structure
import matplotlib.pyplot as plt
import torchvision.utils as utils

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = dataset_3.LoadDataset3(transform=transform, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
testset = dataset_3.LoadDataset3(transform=transform, train=False)
testLoader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True)

#loss function and optimizer
net = drn_structure.resnet50()
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0005)

##train
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        print(i, ' ', loss)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 5))
            running_loss = 0.0

print('Finished Training')

PATH = '../Path/train3_v1.pth'
torch.save(net.state_dict(), PATH)


dataiter = iter(testLoader)
images, labels = dataiter.next()


dataiter = iter(testLoader)
images, labels = dataiter.next()
# print images
plt.imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net1 = drn_structure.resnet50()
net1.load_state_dict(torch.load(PATH))
outputs = net1(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', classes[predicted[j]])
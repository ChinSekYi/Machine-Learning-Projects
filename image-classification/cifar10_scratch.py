import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]  # (mean, sd) for each of the tree RGB colours.
)


batch_size = (
    4  # higher the batch size, the more accurate it is but takes up more memory space
)

num_workers = 2  # if positive int -> Pytorch switch to perform multi-process data loading  # eg. 2 workers simultaneouly putting data into the computer's RAM. Allowd speed up of training process by utilising machines with multiple cores.

# load train data
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)  # every worker will load a whole batch


# load test data
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(trainloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))

print(" ".join("%s" % classes[labels[j]] for j in range(batch_size)))


# Define CNN: models a simple Convolutional NN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # to reshape tensor for our fc # or use x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = Net()

print(net)


# Weights in kernel is learnt during the training process
# Access the weights of the first convolutional layer
conv1_weights = net.conv1.weight.data.numpy()

# Access the weights of the second convolutional layer
conv2_weights = net.conv2.weight.data.numpy()

# Define a loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)

# start.record()
for epoch in range(
    2
):  # an epoch is one pass over the entire train set. Too many epochs may lead to overfitting
    running_loss = 0.0
    for i, data in enumerate(
        trainloader, 0
    ):  # i -> batch number, data = [inputs, labels]
        inputs, labels = data
        optimizer.zero_grad()  # to ensure gradients from multiple passes dont accumulate
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # get derivative of the Loss function for each weight (for each layer) by performing backpropagation
        # propagate derivates backwards using chain rule.
        # values are accumulated in the grad attribute.
        optimizer.step()  # iterates over all the parameters and updates their values (w0 = w1 + lr * grad)

        running_loss += loss.item()

        if i % 2000 == 1999:  # mini-batch size is 2000
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

# end.record()

# waits for everything to finish running
torch.cuda.synchronize()
print("Finished Training")

correct = 0
total = 0
running_corrects = 0

with torch.no_grad():  # used when we dont require Pytorch to run its autograd engine/calculate the gradients of our input.
    for i, data in enumerate(testloader):
        images, labels = data
        outputs = net(images)  # gives the output tensor
        _, predicted = torch.max(
            outputs.data, 1
        )  # _ gives the index and predicted gives the values of the max element
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy of the network on the 10000 test images: %d", 100 * correct / total)


"""
x = torch.tensor(2.0, requires_grad=True)
print(x)
y = x**2
z = y + 2

z.backward()  # dz/d
x.grad

y = x**2
z = y + 2
z.backward()  # dz/d
x.grad
"""

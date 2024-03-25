import matplotlib.pyplot as plt 
import numpy as np 
import os 

import torch 
import torchvision 
from torchvision import datasets, models 
import torchvision.transforms as transforms 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

data_transforms = { 

    'train': transforms.Compose([ 

        transforms.RandomResizedCrop(224), 

        transforms.RandomHorizontalFlip(), 

        transforms.ToTensor(), 

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 

    ]), 

    'val': transforms.Compose([ 

        transforms.Resize(256), 

        transforms.CenterCrop(224), 

        transforms.ToTensor(), 

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 

    ]), 

} 

  
batch_size = 424 
num_workers = 4 

data_dir = "/.../tutorial/hymenoptera_data" 

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
                                          data_transforms[x]) 
                  for x in ['train', 'val']} 

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, 

                                             shuffle=True, num_workers=num_workers) 
              for x in ['train', 'val']} 

trainset = dataloaders['train'] 

testset = dataloaders['val'] 

  

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']} 

  

class_names = image_datasets['train'].classes #['ants', 'bees'] 

  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

 

 

 

 

def imshow(inp, title=None): 

    """Display image for Tensor.""" 

    inp = inp.numpy().transpose((1, 2, 0)) 

    mean = np.array([0.485, 0.456, 0.406]) 

    std = np.array([0.229, 0.224, 0.225]) 

    inp = std * inp + mean 

    inp = np.clip(inp, 0, 1) 

    plt.imshow(inp) 

    if title is not None: 

        plt.title(title) 

    plt.pause(0.001)  # pause a bit so that plots are updated 

  

  

# Get a batch of training data 

inputs, classes = next(iter(dataloaders['train'])) 

  

# Make a grid from batch 

out = torchvision.utils.make_grid(inputs) 

  

imshow(out, title=[class_names[x] for x in classes]) 

 

 

 

 

 

 

#Define CNN: models a simple Convolutional NN 

  

class Net(nn.Module): 
    def __init__(self): 
        super(Net, self).__init__() 
        self.conv1 = nn.Conv2d(3,6,5) 
        self.pool = nn.MaxPool2d(2,2)  
        self.conv2 = nn.Conv2d(6,16,5) 
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84) 
        self.fc3 = nn.Linear(84, 10) 

    def forward(self, x): 
        x = self.pool(F.relu(self.conv1(x))) 

        x = self.pool(F.relu(self.conv2(x))) 

        x = x.flatten(1) 

        #x = x.view(-1, 16 * 5* 5) #to reshape tensor for our fc # or use x.flatten(1) 

        x = F.relu(self.fc1(x)) 

        x = F.relu(self.fc2(x)) 

        x = self.fc3(x) 

        return x  

     

net = Net() 

print(net) 

 

 

 

#Define a loss function and optimiser 

criterion = nn.CrossEntropyLoss() 

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 

 

 

 

for epoch in range(2):  

  

    running_loss = 0.0 

    for i, data in enumerate(trainset, 0): 

        inputs, labels = data  

        optimizer.zero_grad()  

  

        outputs = net(inputs) 

        loss = criterion(outputs, labels) 

  

        loss.backward()   

        optimizer.step() 

  

        #print statistics 

        running_loss += loss.item() 

        """ 

        if i % 2000 == 1999: #mini-batch size is 2000 

            print('[%d, %5d] loss: %.3f' % 

                  (epoch + 1, i + 1, running_loss / 2000)) 

            running_loss = 0.0 

        """ 

  

  

# waits for everything to finish running 

torch.cuda.synchronize() 

  

print('Finished Training') 
 

 

 

 
 
correct = 0 

total = 0 

running_corrects = 0 

  

with torch.no_grad(): #used when we dont require Pytorch to run its autograd engine/calculate the gradients of our input. 

    for i, data in enumerate(testset): 

        images, labels = data 

        outputs = net(images) #gives the output tensor 

        _, predicted = torch.max(outputs.data, 1) #_ gives the index and predicted gives the values of the max element 

  

        total += labels.size(0) 

        correct += (predicted == labels).sum().item() 

  

  

print("Accuracy of the network on the 10000 test images: %d", 100 * correct / total) 

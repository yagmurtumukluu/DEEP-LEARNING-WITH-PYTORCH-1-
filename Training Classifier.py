#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[3]:


# define network


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # dimension: 5x5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        
        # Max pooling 
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        
        
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    

net = Net()
net


# In[7]:


#forward function tanımla yeter.
#backward autograd ile kendiliğinden tanımlanır

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight


# In[11]:


#32x32'lik input: (use MNIST DATASET)

input = torch.randn(1, 1, 32, 32)
out = net(input)
out


# In[19]:


#LOSS FUNCTION

output = net(input)
target = torch.randn(10)  # a dummy target, for example

target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)

loss


# In[21]:


print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


# In[ ]:


#BACKPROP---> To backpropagate the error all 


# In[22]:


net.zero_grad()     

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# In[23]:


#UPDATE THE WEIGHTS

#SGD-----> weight = weight - learning_rate * gradient

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)


# In[25]:


import torch.optim as optim

#create your optimizer

optimizer = optim.SGD(net.parameters(), lr=0.01)

#training loop:


optimizer.zero_grad()   
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() 


# In[ ]:


#TRAIN A CLASSIFIER


# In[26]:


import torch
import torchvision
import torchvision.transforms as transforms


# In[27]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# define batch size
batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[28]:


#SHOW

import matplotlib.pyplot as plt
import numpy as np


# In[35]:


def imshow(img):
    img = img / 2 + 0.5     
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# In[36]:


#random train image


dataiter = iter(trainloader)
images, labels = dataiter.next()


# In[37]:


imshow(torchvision.utils.make_grid(images))

print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


# In[38]:


#Define CNN


# In[39]:


import torch.nn as nn
import torch.nn.functional as F


# In[40]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


# In[42]:


#define loss and optimizer

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[43]:


#train

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        inputs, labels = data

        
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        
        running_loss += loss.item()
        if i % 2000 == 1999:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


# In[44]:


PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)


# In[48]:


#TEST

dataiter = iter(testloader)
images, labels = dataiter.next()



imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[49]:


outputs = net(images)


# In[53]:



#sorrr
#class enerjisi ne kadar yuksek olursa network o kadar fazla görüntünün belirli bir sınıfa ait olduğunu düşünür

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


# In[59]:


#check the perform on the whole data

correct = 0
total = 0

with torch.no_grad(): #gradları sayma
    for data in testloader:
        images, labels = data
        
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: %d %%' % (100 * correct / total))


# In[ ]:





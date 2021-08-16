#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch
import numpy as np


# In[9]:


data = [[1,2], [3,4]]

x_data = torch.tensor(data)


# In[10]:


np_array = np.array(data)
x_np=torch.from_numpy(np_array)


# In[13]:


#Yeni tensör, açıkça geçersiz kılınmadıkça, argüman tensörünün özelliklerini (şekil, veri türü) korur.
x_ones = torch.ones_like(x_data) # x_data'nın özellikelrini korur
print(f"Ones Tensor: \n {x_ones} \n")


x_rand = torch.rand_like(x_data, dtype = torch.float) #x_data türünü geçersiz kılar

print(f"Random Tensor: \n {x_rand} \n" )


# In[20]:


#dimensions 

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


# In[26]:


#Tensor Attributes

tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


# In[27]:


#Tensor Operations


# In[41]:


#indexing and slicing:

tensor = torch.ones(3,3)
tensor [:,2]=8
tensor [1, :]=8
tensor


# In[42]:


#concatenate a sequence of tensors along a given dimension 
#(.cat)

t1=torch.cat([tensor, tensor, tensor], dim=1)
t1


# In[43]:


#multiplying

print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")

# or:

print(f"tensor * tensor \n {tensor * tensor}")


# In[46]:


#matrix multiplication 

print(f" tensor.matmul(tensor)\n {tensor.matmul(tensor.T)} \n ")

print(f"tensor @ tensor.T \n {tensor @ tensor.T}")


# In[47]:


#add

print(tensor, "\n")
tensor.add_(5)
print(tensor)


# In[51]:


#BRIDGE WITH NUMPY

t = torch.ones(3)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")


# In[58]:


#bridge var yani ekleye ekleye artyor dikkat et

t.add_(8)
t


# In[89]:


#NUMPY ARRAY TO TENSOR

n=np.ones(4)
t = torch.from_numpy(n)

#numpy array'de olan degisiklikler 

np.add(n, 8, out=n)
print(f"t: {t}")
print(f"n: {n}")


# In[90]:


#TORCH.AUTOGRAD

#sinir ağı eğitimine güç veren PyTorch'un otomatik farklılaştırma motorudur


# In[91]:


#önceden eğitilmiş bir resnet18 modeli yüklüyoruz. 
#3 kanallı, yükseklik ve genişlik 64 olan tek bir görüntüyü temsil etmek için
#rastgele bir veri tensörü oluşturuyoruz

import torch, torchvision

model = torchvision.models.resnet18(pretrained=True) #pretrained: onceden egitilmis

data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)


# In[92]:


#Forward pass

prediction = model(data)


# In[93]:


#backward

loss = (prediction - labels).sum()
loss.backward()


# In[94]:


#optimizer load

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

#lr =learning rate

optim.step()


# In[95]:


#Differentiation in Autograd

import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)


# In[100]:


Q = 3*a**3 - b**2


# In[106]:


external_grad = torch.tensor([1., 1.])
Q.backward.backward(external_grad = torch.tensor([1., 1.]), retain_graph=True )


# In[107]:


#üst cell çalıştır önce

print(9*a**2 == a.grad)
print(-2*b == b.grad)


# In[108]:


#Exclusion from the DAG

x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does `a` require gradients? : {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")


# In[109]:


#önceden eğitilmiş bir ağın ince ayarının yapılmasıdır.(DAG)

from torch import nn, optim

model = torchvision.models.resnet18(pretrained=True)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False


# In[110]:


#10 label olan dataset'e finetune
model.fc = nn.Linear(512, 10) 


# In[111]:


optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)


# In[ ]:





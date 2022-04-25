#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell 
InteractiveShell.ast_node_interactivity = "all"


#ייבוא ספריות בשביל יצירת הדאטאסט 
from __future__ import print_function, division
import os
import torch
from PIL import Image
import pandas as pd
from skimage import io, transform
import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2
import random

# ייבוא ספריות בשביל למידה
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.autograd import Variable
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
from torchvision.utils import make_grid
from tqdm.auto import tqdm


#הגדרת המכשיר עליו אני רץ
device = 'cpu'
classes = ['Flex', 'Smiley', 'Ghost', 'Heart', 'Thumbs Up', 'Blank']


# In[2]:


class EmojiData(Dataset):
    def __init__(self, setname, root_dir):
        self.setname = setname
        assert setname in ['train','val','test']
        #Define dataset
        overall_dataset_dir = root_dir
        self.selected_dataset_dir = os.path.join(overall_dataset_dir,setname)
        self.all_filenames = os.listdir(self.selected_dataset_dir)
        self.all_labels = pd.read_csv(os.path.join(overall_dataset_dir,'Shwit.csv'),header=0,index_col=0)
        self.label_meanings = self.all_labels.columns.values.tolist()
    
    def __len__(self):
        return len(self.all_filenames)
        
    def __getitem__(self, idx):
        selected_filename = self.all_filenames[idx]
        imagepil = cv2.imread(os.path.join(self.selected_dataset_dir,selected_filename))
        imagepil = cv2.bitwise_not(imagepil)
        M = np.float32([[uniform(.9,1.1),uniform(-.1,.1),uniform(-5,5)],
                        [uniform(-.1,.1),uniform(.9,1.1),uniform(-5,5)],
                       [0,0,1]])
        rows, cols, dim = imagepil.shape
        translated_img = cv2.warpPerspective(imagepil, M, (cols, rows))
        translated_img = cv2.resize(cv2.bitwise_not(translated_img),(48,44))
        translated_img_gray = cv2.cvtColor(translated_img, cv2.COLOR_BGR2GRAY)
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(translated_img_gray)
        label = (torch.Tensor(self.all_labels.loc[selected_filename,:].values)).long()
        sample = {'data':image,
                  'label':label}
        return sample


# In[3]:


train_dataset = EmojiData('train', r"C:\Users\alleh\DataSheet")
val_dataset = EmojiData('val', r"C:\Users\alleh\DataSheet")
loaders = {
    'train' : DataLoader(train_dataset, batch_size= 5, shuffle=True),
    
    'val'  : DataLoader(val_dataset, batch_size=1, shuffle=True),
}


# In[4]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(1152, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


# In[5]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[6]:


running_loss = 0
#check = transforms.Compose([transforms.ToPILImage()])
for epoch in range(600):  # loop over the dataset multiple times
    for i, data in enumerate(loaders['train'], 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['data'], data['label'].flatten()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()  
    if(epoch%100 == 0 and epoch > 0):
        print(loss)
print(loss)
print('Finished Training')


# In[7]:


PATH = './SavedModel.pt'
torch.save(net.state_dict(), PATH)


# In[8]:


net = Net()
net.load_state_dict(torch.load(PATH))


# In[9]:


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in loaders['val']:
        images, labels = data['data'], data['label']
        # calculate outputs by running images through the network
        outputs = net(images)
        #print(outputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        print("The prediction is: ", predicted, " And the True Label is:", labels)
        #print(predicted, labels)
        #print("Prediction is: ",predicted,", And True Label is: ", labels)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 100 test images: {100 * correct // total} %')


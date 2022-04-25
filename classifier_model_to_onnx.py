#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


class EndNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) # by now we got depth 16 8x9 network
        self.fc1 = nn.Linear(1152, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.reshape(480, 440, 4)
        x = torch.narrow(x, dim = 2, start = 3, length = 1) # we need the "non-transparent", so we take the alpha channel
        x = x.reshape(1, 1, 480, 440)
        x = F.avg_pool2d(x, 10, stride=10) # reducing the image to the net size
        x = x/255 # moving from uchar encoding to float encoding
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[3]:


pytorch_model = EndNet()
pytorch_model.load_state_dict(torch.load(r"C:\Users\alleh\SavedModel.pt"))
pytorch_model.eval()
dummy_input = torch.zeros(480 * 440 * 4)
torch.onnx.export(pytorch_model, dummy_input, 'onnx_model.onnx', verbose=True)


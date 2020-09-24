import os
import sys
import json
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import copy
from PIL import Image



fn = sys.argv[1]
if os.path.exists(fn):
    test_dir = os.path.basename(fn)

mapping = {
    0: 'male',
    1: 'female'
}


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, 7, stride=4, padding=1)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)

        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)

        self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.norm3 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)

        self.fc1 = nn.Linear(18816, 512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, 10)

        self.apply(weights_init)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)

        x = nn.functional.leaky_relu(self.conv2(x))
        x = self.pool2(x)
        x = self.norm2(x)

        x = nn.functional.leaky_relu(self.conv3(x))
        x = self.pool3(x)
        x = self.norm3(x)

        x = x.view(-1, 18816)

        x = self.fc1(x)
        x = nn.functional.leaky_relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = nn.functional.leaky_relu(x)
        x = self.dropout2(x)

        x = nn.functional.log_softmax(self.fc3(x), dim=1)

        return x

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=1e-2)

def test_on_a_class(image_tensor):
    with torch.no_grad():
        net = Net()#.to(device)
        net.load_state_dict(torch.load(f'{PATH_TO_MODELS}/{filename}.pt'))
        net.eval()
        output = net(image_tensor)
        output = torch.max(output, 1)[1]#.to(device)
        if output.item() == 0:
            result = 'male'
        else:
            result = 'female'
        #result = f"{mapping[output.item()]}"

    return result

def test(path):
    image = Image.open(path)
    plt.imshow(image)
    trans = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(227),
                transforms.ToTensor()
            ]
    )
    image = trans(image)
    image.unsqueeze_(0)
    #image = image.to(device)
    return test_on_a_class(image)

PATH_TO_MODELS = './'
filename = 'my_model'
json_obj = {}
test_files = os.listdir(test_dir)
for file in test_files:
    json_obj.update({file: test(f'{test_dir}/{file}')})

with open('process_results.json', 'w') as jsonFile:
    json.dump(json_obj, jsonFile)

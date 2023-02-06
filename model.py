import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import os
import scipy.io
import torch
import numpy as np

net = nn.Sequential(
    nn.Conv2d(3, 32, 3),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3),
    nn.ReLU(),
    nn.MaxPool2d(2, 2, 3),
    
    nn.Conv2d(64, 128, 3),
    nn.ReLU(),
    nn.Conv2d(128, 256, 3),
    nn.ReLU(),
    nn.MaxPool2d(2, 2, 3),


    nn.Conv2d(256, 512, 3),
    nn.ReLU(),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.MaxPool2d(2, 2, 3),
    

    nn.Flatten(),
    nn.Linear(884736, 10000),
    nn.ReLU(),
    nn.Linear(10000, 500),
    nn.ReLU(),
    nn.Linear(500, 2),
    
)
              
class GazeTrackingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(GazeTrackingDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for label in os.listdir(self.root_dir):
            if str(label)[0] != ".":
                path = os.path.join(self.root_dir, label)
                for image_name in os.listdir(path):
                    image_path = os.path.join(path, image_name)
                    mat = scipy.io.loadmat(image_path)

                    # Extract the data from the .mat file
                    data = mat['data']

                    # Convert the data to a numpy array
                    data = np.array(data)

                    # Convert the numpy array to a PIL image
                    image = Image.fromarray(data)

                    # Convert the PIL image to a PyTorch tensor
                    tensor_image = torch.from_numpy(np.array(image))
                    self.images.append(tensor_image)
                    self.labels.append(image_name)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label
    

gaze_dataset = GazeTrackingDataset("../../gaze-dataset/Data/Normalized")
gaze_dataloader = DataLoader(gaze_dataset, batch_size=32)

gaze_img, _ = next(iter(gaze_dataloader))

plt.imshow(gaze_img)

print(net)
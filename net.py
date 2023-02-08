import torch.nn as nn
import torch

INPUT_IMAGE_HEIGHT = 224
INPUT_IMAGE_WIDTH = 224


from vgg import vgg

            


class GazePredictor(nn.Module):
    def __init__(self):
        super(GazePredictor, self).__init__()
        self.image_net = nn.Sequential(
            nn.Conv2d(512, 32, 3),
            nn.ReLU(),
            nn.AvgPool2d(3),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.AvgPool2d(3),
            nn.Flatten(),
            nn.Linear(256, 200),
            nn.ReLU(),
        )
        self.face_point_net = nn.Sequential(
            nn.Conv1d(3, 64, 3),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(60416, 1000, 3), 
            nn.ReLU(),
        )

        self.vgg = vgg
        self.vgg.load_state_dict(torch.load("./vgg_normalised.pth"))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
        
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.final_processing = nn.Sequential(
            nn.Linear(1200, 500),
            nn.ReLU(),
            nn.Linear(500, 2)
        )

    def forward(self, image, face_point):
        image_encoded = self.image_net(self.vgg(image))
        face_encoded = self.face_point_net(face_point)
        combined_encoding = torch.cat((image_encoded, face_encoded), 1)

        output = self.final_processing(combined_encoding)

        return output


import torch
import torch.nn as nn
import torchvision.models as models


class AlexNet(nn.Module):
    def __init__(self, num_classes=3):
        super(AlexNet, self).__init__()
        pre_trained_backbone = models.alexnet(pretrained=True)
        alex_features = pre_trained_backbone.features
        self.features = nn.Sequential(*list(
                    alex_features.children())[:-1])
        self.fc = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Linear(256*3*3, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 256),
                                nn.ReLU(),
                                nn.Linear(256, num_classes))

    def forward(self, x):
        # turns off the backwards gradient to the feature extractor backbone
        with torch.no_grad():
            x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    alexnet = AlexNet(3)
    print(alexnet)
